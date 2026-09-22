# Transaction Analysis Agent

A GenAI data-analysis agent that answers natural-language questions about a
transaction dataset — where the language model decides **which operations** to
run and Python does **all** of the arithmetic.

```
> What is the total revenue for UK transactions?

Answer:
Sum of revenue for region UK: 4,320.00

Operations performed:
1. Filtered transactions where region = UK (4 of 10 rows kept).
2. Computed revenue for each of the 4 matching transactions (revenue = units * unit_price * (1 - discount)).
3. Summed revenue over 4 matching transactions.

Explanation:
The result was computed deterministically in Python from the active dataset
(project_4.csv); the language model only chose which operations to run.
```

## The idea

The boundary between probabilistic reasoning and deterministic computation is
the whole design:

```
question
  -> pre-screen        deterministic   obvious out-of-scope requests stop here, before any API call
  -> planner (LLM)     probabilistic   natural language -> a structured analysis plan (JSON)
  -> plan schema       deterministic   Pydantic: are these real tools with allowed options?
  -> validator         deterministic   do these names and values exist in THIS dataset?
  -> executor          deterministic   six pandas functions, dispatched by step type
  -> renderer          deterministic   answer + operations performed + explanation
```

The model never sees a transaction row, never receives a file path, and never
produces a number that appears in an answer. It emits a plan like this:

```json
{"status": "success",
 "intent": "Sum of revenue for region UK",
 "steps": [{"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
           {"tool": "compute_metric", "metric": "revenue"},
           {"tool": "aggregate", "column": "revenue", "func": "sum"}]}
```

There is no `execute_python` tool, no formula string, no file path — anywhere
in the tool interface. A test enforces that the `app/` package contains no
`eval(`, `exec(`, `compile(`, `__import__`, `subprocess`, `os.system`,
`pickle` or `shutil`, and that only the data loader opens files.

## Quick start

Requires Python 3.12+ and a Gemini API key
([free tier](https://aistudio.google.com/apikey), no card needed).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

cp .env.example .env            # then put your key in GEMINI_API_KEY

python run.py --data data/project_4.csv
```

A session looks like this:

```
Loaded project_4.csv: 10 transactions.
  Columns: id, date, region, product, units, unit_price, discount
  Dates:   2026-01-03 to 2026-03-04
  Missing: none
  The file also carries 10 questions (/questions).
Ready. Ask a question, or /help for commands.

> Which region has the highest total revenue?
> How many transactions have a missing discount?
> /load path/to/another.csv
> /exit
```

Commands: `/help`, `/profile`, `/questions`, `/load <path>`, `/exit`. The
dataset is loaded once and every question runs against it until you `/load`
another.

## What it can answer

| Kind | Example |
|---|---|
| Aggregate | "What is the total revenue for UK transactions?" |
| Average / median | "What is the average number of units per transaction?" |
| Count | "How many Beta transactions are there?" |
| Grouped statistic | "What is the median revenue by region?" |
| Extreme | "Which region has the highest total revenue?" |
| Grouped filter | "Which regions have more than 5 transactions?" |
| Date range | "What was the revenue between January 1 and February 15?" |
| Missing values | "How many transactions have a missing discount?" |
| Numeric comparison | "How many transactions have more than 5 units?" |

These are examples, not a list of supported sentences. Nothing in the code
keys off question wording.

### The tool interface

Six operations, each with a closed set of options:

| Tool | Options |
|---|---|
| `filter_rows` | `eq, neq, gt, gte, lt, lte, between, in, is_null, not_null` |
| `compute_metric` | `revenue` |
| `aggregate` | `sum, mean, median, min, max, count` |
| `group_by` | key: `region`, `product`; same aggregations; no column needed for `count` |
| `filter_groups` | `eq, neq, gt, gte, lt, lte, between` — applied to a group's aggregate (SQL `HAVING`) |
| `select_extreme` | `highest`, `lowest` — ties are all reported, never resolved arbitrarily |

## Data contract

Required columns: `id`, `date` (ISO `YYYY-MM-DD` only), `region`, `product`
(categorical), `units`, `unit_price`, `discount` (numeric; a **fraction**, so
`0.20` means 20 %). A file missing any of them is rejected.

Derived metric:

```
revenue = units * unit_price * (1 - discount)
```

**The `question` column.** The supplied file interleaves transaction rows with
evaluation questions. A non-empty `question` cell is the *only* thing that
marks a row as a question; a transaction with blank fields is still a
transaction. Question rows are excluded from analysis and kept as a corpus
(`/questions`).

**Extra columns** are loaded and reported, but are not analysable: existing in
the file is not the same as being part of the supported semantic contract.

**Nothing about the data is in the code.** Category values, ranges, dates and
row counts are discovered from whichever file is loaded. Swap in a CSV with
different regions and products and the same binary answers questions about it —
and correctly refuses values that no longer exist.

### Missing values

* Aggregations skip missing values and report how many rows were usable:
  *"Computed from 9 of the 10 matching transactions; 1 had a missing value and
  was left out rather than treated as zero."*
* `count` counts rows, not non-null values.
* A derived metric is missing whenever any input is missing. Nothing is
  zero-filled.
* Missing values never satisfy a comparison: a row with no `units` is in
  neither `units > 5` nor `units <= 5`.
* Rows whose group key is missing belong to no group; they are excluded and
  counted.
* A value that was present but unparseable (`"ten"`, `"Jan 3rd"`) becomes
  missing *and* is counted separately from genuinely empty cells.
* Nothing to compute from is reported as **no data**, never as zero.

## Guardrails

| Situation | Behaviour |
|---|---|
| Fabricated column or metric (`profit`, `GST`) | Rejected, naming what is supported |
| Unsupported aggregation | Rejected — never silently substituted |
| Unknown category value (`Britain`) | Rejected with the real values; synonyms are never mapped |
| Near-miss value (`Gama`) | "Did you mean 'Gamma'?" |
| Ambiguous wording ("in Mars", "discount over 10") | Clarification requested, not guessed |
| Code execution / file access / secrets | Rejected before the model is called |
| Rephrased attack ("act as a terminal") | Reaches the planner by design, and is refused there |
| A jailbroken model returning `{"tool": "run_shell"}` | No such tool exists — the plan fails validation and nothing runs |
| Dates outside the data | A valid question: executed, answered "no data", with the dataset's real range |

Safety does not rest on the pre-screen or on the model behaving. It rests on
there being no expressible operation outside those six tools.

## Testing

```bash
python -m pytest -W error          # 350 passed, 2 skipped
RUN_LIVE_LLM=1 python -m pytest    # also runs the two tests that call Gemini
```

Every test but two runs without a network or an API key: a `FakeLLMClient`
returns scripted plans and records what it was asked, so a failure means
*Python* misbehaved, not that a model phrased something differently. Guardrail
tests script a *cooperating* model — one that returns exactly what an attacker
asked for — and assert Python refuses anyway.

No expected answer is a literal from the supplied file. End-to-end tests
compute their expectation inside the test with plain pandas over whatever file
is in `data/`, so they keep passing if the data changes — the same property
the application has.

## Layout

```
app/
  contract.py      column roles and derived metrics (fixed)
  data_loader.py   path -> ActiveDataset (the only filesystem touch)
  data_profile.py  the loaded data described at runtime (discovered values)
  tools.py         the six deterministic operations
  schemas.py       the analysis plan; structural validation
  validator.py     semantic validation against the profile
  executor.py      runs a validated plan
  planner.py       question -> plan, via the LLM
  prescreen.py     cheap filter for obvious out-of-scope requests
  agent.py         the pipeline, wired
  renderer.py      answer / operations performed / explanation
  cli.py           the interactive session
  llm/             provider interface, Gemini implementation, test double
data/project_4.csv
tests/             350 tests
run.py
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | required |
| `GEMINI_API_KEY_2` | – | optional second key, tried automatically when the first is rate limited |
| `GEMINI_MODEL` | `gemini-3.6-flash` | planning model |
| `GEMINI_FALLBACK_MODEL` | none | optional second model, tried on a 429 or 503 |
| `LLM_PROVIDER` | `gemini` | the provider sits behind an `LLMClient` interface |

## Limitations

* **Free-tier quota.** Gemini's free tier allows roughly 20 planning requests
  per day per key, after which questions return an honest `Status: Error`
  rather than an answer. Setting `GEMINI_API_KEY_2` doubles the ceiling (the
  client rotates to it on a 429) but does not remove it; enabling billing on a
  key does. A question costs roughly 1,450 tokens.
* Plan quality depends on the model. The architecture guarantees that a bad
  plan is *rejected*, not that every question produces one.
* Answers are single values, grouped values or an extreme — the agent does not
  list matching transactions.
