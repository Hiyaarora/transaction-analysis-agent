# Architecture & Design Decisions

*Transaction Analysis Agent — how it is built, and why it is built this way.*

This document is for someone seeing the project for the first time. It walks
the code in execution order, names every file and what it is responsible for,
and records the decisions that shaped it. The [README](README.md) covers how to
run it.

---

## 1. The thesis

> **The language model decides *which operations* to run. Python runs them.**

Every design choice follows from that one sentence. An LLM is excellent at
turning "which region did best?" into "group by region, sum revenue, take the
highest" — and unreliable at arithmetic, at remembering what is in a file, and
at refusing to invent a plausible-looking number. So the model is given exactly
one job, and it is a job it is good at: translation.

The model never sees a transaction row. It never receives a file path. It never
produces a number that reaches the user. What it produces is a JSON plan:

```json
{"status": "success",
 "intent": "Sum of revenue for region UK",
 "steps": [{"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
           {"tool": "compute_metric", "metric": "revenue"},
           {"tool": "aggregate", "column": "revenue", "func": "sum"}]}
```

Python then checks that plan twice, executes it with pandas, and describes what
it did. If the model returns nonsense, the plan is rejected. If the model is
jailbroken entirely and returns `{"tool": "run_shell", "cmd": "..."}`, there is
no such tool — the plan fails to parse and nothing runs.

**The security property is structural, not behavioural.** It does not depend on
the model behaving, on a system prompt holding, or on a blocklist being
complete. It depends on there being no expressible operation outside six pure
functions.

---

## 2. The pipeline

```
question (natural language)
   │
   ├─ 1. prescreen.py      deterministic   obvious code/file/secret requests stop here
   │                                       (cost saving — NOT the security boundary)
   ├─ 2. planner.py        PROBABILISTIC   the only LLM call in the system
   │        └─ llm/*.py                    provider client → raw JSON text
   ├─ 3. schemas.py        deterministic   Level 1: is this a well-formed plan?
   ├─ 4. validator.py      deterministic   Level 2: does it make sense for THIS data?
   ├─ 5. executor.py       deterministic   dispatch table → tools.py
   │        └─ tools.py                    six pure pandas functions
   └─ 6. renderer.py       deterministic   answer + operations performed + explanation
```

Steps 1 and 3–6 contain no randomness and no network. Step 2 is the only place
uncertainty enters, and it is fenced on both sides.

---

## 3. Walking the code, in order

### 3.1 Two entry points, one engine

| Entry | File | What it does |
|---|---|---|
| Terminal | `run.py` → `app/cli.py` | Loads a CSV once, then a read-ask-print loop |
| Web | `run_api.py` → `app/api/main.py` | Serves the JSON API *and* the built React app |

Both construct the same `Agent`. That is deliberate: two front ends that
answered differently would mean the analysis logic had leaked into one of them.

**Terminal flow**

```
run.py
  └─ app/cli.py : run()
       ├─ app/data_loader.py : load_dataset(path)   → ActiveDataset
       ├─ app/config.py      : load_settings()      → Settings (from .env / environment)
       ├─ app/llm/factory.py : build_llm(settings)  → LLMClient
       ├─ app/agent.py       : Agent(dataset, llm)
       └─ loop: line → agent.ask() → app/renderer.py : render() → stdout
```

**Web flow**

```
browser
  └─ frontend/src/api/client.ts     POST /api/ask  + X-Session-Id header
       └─ app/api/routes.py : ask()
            ├─ app/api/sessions.py  : SessionStore.get(session_id) → that tab's Agent
            ├─ app/agent.py         : Agent.ask(question)
            └─ app/api/convert.py   : AgentResponse → app/api/schemas.py wire model
       └─ frontend/src/hooks/useSession.ts   caches the answer
            └─ frontend/src/components/ResultView.tsx   renders it
```

### 3.2 Loading the data — `app/data_loader.py`

**The only module in the application that opens a data file.** It receives a
path from the CLI or bytes from an upload — never from the model.

Three decisions live here:

1. **Everything is read as a string first** (`dtype=str, keep_default_na=False`),
   then converted column by column. This keeps *empty* and *unparseable*
   distinguishable. A blank discount cell and a discount of `"ten"` are both
   missing for analysis, but they are counted separately and reported
   separately, because they mean different things about the data's quality.
2. **Dates are ISO-8601 only.** Accepting `03/04/2026` would mean guessing
   between March 4th and April 3rd, and a silent wrong guess is worse than a
   rejection.
3. **The `question` column is the row discriminator.** The supplied file
   interleaves transaction rows with evaluation questions. A *non-empty
   `question` cell* is the only thing that marks a row as a question — not
   blank numeric fields, not a missing id. Guessing any other way would have
   corrupted every count in the dataset.

Output is an `ActiveDataset`: a transactions-only DataFrame, the question
corpus, the source name, and a `DataProfile`.

### 3.3 Two kinds of schema — `app/contract.py` and `app/data_profile.py`

This split is the reason the project can honestly claim "nothing about the data
is hardcoded".

| | `contract.py` | `data_profile.py` |
|---|---|---|
| Holds | Which columns must exist, what role each plays, the revenue formula | Which regions/products actually occur, date span, numeric ranges, missing counts |
| Known | At design time | At load time |
| Changes when | The *domain* changes | A different *file* is loaded |

`revenue = units * unit_price * (1 - discount)` is in `contract.py` because it
is a definition. `region ∈ {UK, DE, FR}` is in the profile because it is an
observation. Load a CSV with Japanese regions and Korean products and the same
binary answers questions about it — and correctly refuses values that no longer
exist.

### 3.4 The pre-screen — `app/prescreen.py`

Stops requests that are unmistakably system-level ("run python", "cat
/etc/passwd", "print your API key") before an API call is made.

**This is explicitly *not* the security boundary.** It is a cost optimisation:
it saves ~1,300 prompt tokens, the latency, and a slot in a free-tier daily
quota. It is deliberately kept narrow and high-confidence rather than being
grown into a blocklist, because a blocklist that is *believed* to be the
boundary is more dangerous than no blocklist at all — it invites relaxing the
things that actually hold. A rephrased attack ("act as a Linux terminal")
sails past it by design, reaches the planner, and is refused there and at
validation.

### 3.5 The planner — `app/planner.py` (+ `app/llm/`)

The only probabilistic step. It builds a prompt from the `DataProfile` and the
tool catalogue, calls the provider, and parses the reply.

**The constructor takes a `DataProfile`, not an `ActiveDataset`.** This is the
context boundary expressed as a type: there is no DataFrame in scope, so no row
can be leaked into a prompt even by accident. A test asserts that the prompt
sent to the provider contains no transaction values and no file path.

The provider sits behind one interface, `app/llm/base.py`:

```python
class LLMClient:
    def complete_json(self, *, system: str, user: str, schema: dict) -> str: ...
```

It returns **raw text**, not a parsed object. Parsing belongs to `schemas.py`,
which validates every reply identically regardless of provider. A provider
module's only extra jobs are to pass the JSON schema for constrained decoding
(an optimisation, never a guarantee) and to translate its own exceptions into
`LLMError`, so no vendor type leaks upward.

### 3.6 Level 1 — `app/schemas.py`

Pydantic models. Is this a *well-formed* plan?

Two features do the heavy lifting:

* **A discriminated union on `tool`.** An unknown tool name fails here. There is
  nothing to dispatch on and nothing to run.
* **`extra="forbid"`.** A smuggled extra field — `{"tool": "aggregate",
  "func": "sum", "cmd": "rm -rf /"}` — is a parse error, not an ignored key.

Every option set (`FILTER_OPS`, `AGG_FUNCS`, …) is imported from `tools.py`, so
there is exactly one list of what is allowed and the schema cannot drift from
the implementation.

If parsing fails, the planner retries **once** with the error text. If it fails
again, the question is refused. There is no third attempt: a model that has
produced malformed JSON twice is not about to succeed on the third try, and the
cost is real.

### 3.7 Level 2 — `app/validator.py`

A well-formed plan can still be nonsense *for this dataset*. This module walks
the steps in order and answers what no schema can:

* **Do these names exist here?** `profit` is a well-formed metric name and not a
  real one. `Britain` is a well-formed region and not one in the file.
* **Is the plan coherent?** You cannot `select_extreme` before you have
  `group_by`. The validator tracks a stage — rows → scalar, or rows → groups →
  extreme — and rejects a sequence that cannot happen.
* **Did the user actually say this value?** See §4.4.

It returns a `ValidationOutcome` — `ok`, `clarification_required`, or
`rejected` — rather than raising or mutating the plan. Three outcomes, three
different things to say to the user.

### 3.8 Execution — `app/executor.py` and `app/tools.py`

The executor is deliberately dumb: a dispatch table from step type to a
function in `tools.py`, plus typed state moving through stages.

```
DataFrame --aggregate------> AggregateResult                                    (scalar)
DataFrame --group_by-------> GroupResult --filter_groups--> GroupResult
                                         --select_extreme-> ExtremeResult
```

`tools.py` is **the whole of what can be executed**: six pure pandas functions,
each with a closed tuple of options.

| Tool | Answers | Options |
|---|---|---|
| `filter_rows` | which rows? | `eq neq gt gte lt lte between in is_null not_null` |
| `compute_metric` | add a derived column | whitelist (`revenue`) — formulas live in Python |
| `aggregate` | one number from a column | `sum mean median min max count` |
| `group_by` | one number per category | key: `region`, `product` |
| `filter_groups` | which groups qualify? | `eq neq gt gte lt lte between` (SQL `HAVING`) |
| `select_extreme` | which group is top/bottom? | `highest`, `lowest` |

There is **no function that accepts an expression, a formula string, a path, or
code**. A caller wanting something outside these options gets a `ToolError`,
not an interpretation. A test asserts the `app/` package contains no `eval(`,
`exec(`, `compile(`, `__import__`, `subprocess`, `os.system`, `pickle` or
`shutil`, and that only the data loader opens files.

### 3.9 Presentation — `app/renderer.py`

Writes the only prose in the system, from facts that were already computed: the
`StepRecord`s the executor emitted and the values in the result objects. **It
has no access to the DataFrame**, so it cannot produce a number Python did not
calculate. Output is always three parts: the answer, the operations performed
(with row counts at each step), and a one-line explanation of where the number
came from.

---

## 4. The decisions worth defending

### 4.1 Why six tools, and why a closed option set

The alternative — one `run_query` tool taking a pandas expression — is far more
flexible and completely indefensible. Flexibility for the model is attack
surface for everyone else. Six tools with enumerated options means the set of
things that can possibly happen is finite, listable, and testable. Every
guardrail test scripts a *cooperating* model — one that returns exactly what an
attacker asked for — and asserts Python refuses anyway.

### 4.2 Why `filter_groups` is a separate tool

"Which regions have more than 5 transactions?" filters *after* aggregation —
SQL's `HAVING`, not `WHERE`. This could have been an optional condition on
`group_by`. It was made a separate tool because the two filter at different
stages over different things (rows vs. one number per group), and merging them
would mean one tool whose meaning depends on where it sits in the plan. Note it
excludes `in`, `is_null` and `not_null`: those make no sense applied to a
group's single aggregate.

### 4.3 The missing-value policy

Every rule here exists because the alternative silently produces a wrong number
that looks right:

* Aggregations **skip** missing values and report `rows_used` of `rows_total`
  — *"computed from 9 of the 10 matching transactions"*. Silently averaging over
  9 while claiming 10 is a lie of omission.
* `count` counts **rows**, not non-null values. "How many transactions have a
  missing discount?" is `filter is_null` then `count`.
* A derived metric is missing if **any** input is missing. Nothing is
  zero-filled — a missing discount is not a 0% discount.
* Missing values satisfy **no** comparison. A row with no `units` is in neither
  `units > 5` nor `units <= 5`.
* Rows with a missing group key belong to no group; they are excluded **and
  counted**.
* Nothing to compute from is reported as **no data**, never as `0`. Zero is an
  answer; no data is the absence of one.

### 4.4 The value-origin guard — the subtlest rule

A model asked "Which region performed best for **Germany**?" may plan
`region = "DE"`. `DE` is a real region, so both validation levels pass — and the
user gets a confident answer to a question they did not ask.

So the validator adds one check: **a category value used in a plan must appear
as a whole word in the user's question.** If it does not, the response is a
clarification listing the real values.

This is deterministic and data-independent. Nothing about "Germany" or "DE" is
in the code; the rule is a comparison between the plan and the question text,
so it holds for any synonym, any language, and any model. Prompt instructions
were tried first and held for only two of three phrasings — a tendency, not a
guarantee.

*Known limitation, accepted deliberately:* a one- or two-letter category code
that is also an English word (a region `IN`, versus the word "in") can satisfy
the guard by coincidence.

### 4.5 Why two providers, and how failure is routed

`LLM_PROVIDER` selects the primary; `LLM_FALLBACK_PROVIDER` names a second.
Three layers, each escaping a different failure:

1. **Inside a provider.** Groq retries once on 429/500/502/503/504. Gemini
   routes by error class: a **429** is the *key's* quota, so it rotates to a
   second key; a **503** is the *model* being saturated, which a second key
   cannot escape, so it rotates to a second model. A uniform retry chain was
   the first attempt and was wrong — these are opposite escapes.
2. **Across providers** (`app/llm/chain.py`). Exactly two attempts, one per
   provider, for the case a provider is down or out of quota for the day. Groq
   and Gemini fail for unrelated reasons, so one being unavailable says little
   about the other.
3. **To the user.** If both fail, the response is an honest `error` status — an
   HTTP **200** with `status: "error"`, not a crash. The session stays usable
   and the question can be asked again.

**Why Groq is the default:** measured on the ten questions the dataset carries,
with every answer checked against a value computed independently in pandas —
Groq `openai/gpt-oss-120b` at a **1.2 s median, 10/10 correct**; Gemini
`gemini-3.6-flash` at **30.2 s, 10/10 correct**. Both plan correctly; one is
twenty-five times quicker. (The Gemini latency was itself a finding: the 2.x
`thinking_budget` parameter is ignored by 3.x models, which then reason at full
depth. The 3.x parameter is `thinking_level`.)

Validation and execution together take **under 10 ms**. Effectively all latency
is the provider's, which is why provider choice — not optimisation — was the
lever that mattered.

### 4.6 Sessions, and why they are in memory

A session is a UUID the browser generates on page load, sent as `X-Session-Id`.
It is held in React state — deliberately *not* `localStorage`, so a refresh
starts clean rather than referring to a dataset the server may no longer have.

Server-side, `app/api/sessions.py` is a dict with a TTL and a cap, so a
long-running process cannot accumulate DataFrames. The four-method interface
means replacing it with Redis is a single-file change. It is in memory because
this is a single-instance demo — and because pretending otherwise would be the
kind of unused abstraction that is worse than none.

### 4.7 Uploads never become a path

The browser sends **bytes**. The server chooses its own temporary filename,
hands it to the loader, and deletes it in a `finally` — after failure as well as
success. The user's filename is used only as a label, stripped of any directory
component, so `../../etc/evil.csv` becomes `evil.csv`.

A failed upload leaves the previous dataset active. The original bytes stay in
the session so the file can be downloaded back unchanged; nothing re-reads them
for analysis.

### 4.8 Caching answers in the browser

Answers are cached per `dataset + question`, so paging back and forth between
questions costs no API calls. **Errors are never cached** — a clarification or a
rejection is determined by the question and the data and will not change, but an
error is a moment in time, and asking again is the entire point of asking again.
Replacing the dataset changes the cache key prefix, so one file's numbers can
never be shown against another file's profile.

### 4.9 Testing strategy

475 Python tests and 62 frontend tests, written alongside each phase rather than
after it. Two properties matter more than the count:

* **Almost every test runs with no network and no API key.** A `FakeLLMClient`
  returns scripted plans and records what it was asked. So a failing test means
  *Python* misbehaved — not that a model phrased something differently today.
  It also makes the *prompt* testable: a test asserts what the planner was sent.
* **No expected answer is a literal from the supplied file.** End-to-end tests
  compute their expectation inside the test with plain pandas over whatever is
  in `data/`. They keep passing if the data changes — the same property the
  application has.

---

## 5. Deployment shape

One Render web service. The FastAPI process serves `/api/*` and, from the same
origin, the React production build in `frontend/dist`.

```
browser ──▶ https://<service>/            index.html + hashed assets
            https://<service>/api/ask     the agent
```

One origin is the point: the browser never makes a cross-origin request, so
there is **no CORS configuration in production**, and the API key stays in the
server process. The React bundle contains no key, no SDK and no provider logic —
verified by grepping the built output.

Route order matters. The API router is registered first so no static route can
shadow an endpoint; the catch-all then falls back to `index.html` for client
routes, **except** under `/api`, where a missing endpoint stays a JSON 404 rather
than becoming HTML the client cannot parse.

Known characteristics of a free single-instance demo, each a deliberate trade:
the service **sleeps** after ~15 minutes idle (30–60 s cold start, which the UI
explains rather than spinning silently); **sessions do not survive a restart**;
**uploads are not persisted**; and it runs **one worker**, because sessions are
in-process and a second would answer half of a user's requests with "no
dataset".

---

## 6. What I would do next

* **Move session state out of the process** (Redis, or a dataset token the
  client presents each request) — the one change that unlocks horizontal
  scaling.
* **Cache plans, not just answers.** Two phrasings of the same question produce
  the same plan; a normalised plan cache would cut provider calls further.
* **Widen the tool interface carefully** — a `top_n` tool and multi-column
  grouping are the two gaps real users hit first. Each addition is a new closed
  option set, a schema change and a validator rule, which is exactly the cost it
  should have.
* **A smaller planning model.** Planning is translation, not reasoning; the
  120B model is almost certainly larger than the job requires.
