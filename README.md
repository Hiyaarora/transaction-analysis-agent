# Transaction Analysis Agent

An agentic data-analysis tool that answers natural-language questions about a
transaction dataset — where a language model decides **which operations** to
run and Python does **all** of the arithmetic.

**Live demo:** _(add the Render URL here once deployed)_

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

The model never sees a transaction row, never receives a file path, and never
produces a number that reaches you. It returns a plan; Python validates it
twice, executes it with pandas, and reports what it did.

📄 **[ARCHITECTURE.md](ARCHITECTURE.md)** — the full walkthrough: every file, the
execution flow, and the reasoning behind each decision.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12, FastAPI, pandas, Pydantic |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS v4 |
| LLM (default) | **Groq** — `openai/gpt-oss-120b` |
| LLM (fallback) | **Gemini** — `gemini-3.6-flash` |
| Tests | pytest (475), Vitest (62) |
| Deployment | Render — one web service, FastAPI serves the API and the built UI |

**Why Groq is the default:** on the ten questions the dataset carries, with every
answer checked against pandas — Groq answered 10/10 at a **1.2 s median**, Gemini
10/10 at **30.2 s**. Gemini is asked only when Groq cannot answer at all (out of
quota, or unreachable). Validation and execution take under 10 ms; effectively
all latency is the provider's.

---

## Running it

### 1. Setup

Needs Python 3.12 and a **Groq API key**
([free, no card](https://console.groq.com/keys)).

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

cp .env.example .env             # then put your key in GROQ_API_KEY
```

> On Windows, if `activate` is blocked by the execution policy, just call the
> interpreter directly: `.venv\Scripts\python.exe run.py ...`

### 2. In the terminal

```bash
python run.py --data data/project_4.csv
```

```
> Which region has the highest total revenue?      ask anything
> /questions                                       list the questions in the file
> /ask 3                                           ask the third one
> 3                                                the same, typed faster
> /profile                                         what was discovered in the data
> /load path/to/another.csv                        switch datasets
> /exit
```

`Thinking / 4s` appears on one line while the model plans, and is erased before
the answer.

### 3. In the browser

Two terminals — the API, and the Vite dev server which proxies `/api` to it:

```bash
python run_api.py                             # terminal 1 → http://127.0.0.1:8000
cd frontend && npm install && npm run dev     # terminal 2 → http://localhost:5173
```

Open **http://localhost:5173**.

To run it exactly as deployed — one port, no Vite:

```bash
cd frontend && npm run build && cd ..
python run_api.py                             # http://localhost:8000 serves both
```

### 4. Tests

```bash
python -m pytest                   # 475 passed, 2 skipped
cd frontend && npx vitest run      # 62 passed
```

---

## What you can do with it

**Use the supplied dataset.** `data/project_4.csv` carries ten evaluation
questions inside the file itself. Step through them in the UI, or `/ask 1` in
the terminal.

**Upload your own CSV.** Any file with the required columns works — see the data
contract below. Two cases, both handled:

| Your file | What happens |
|---|---|
| **With** a `question` column | Those rows are set aside as a question list and shown for you to step through. They are excluded from every calculation. |
| **Without** questions | Loads normally; the question list is simply empty. Ask your own. |

**Ask anything you like.** Free-text questions go to the same engine as the
built-in ones. Nothing in the code keys off question wording.

**Swap in different data.** Same columns, different regions, products, dates and
numbers — the same binary answers questions about it, because category values,
ranges and row counts are all discovered at load time. It will also correctly
refuse values that no longer exist in your file.

**Download what is loaded.** The dataset tile is a download button, so you can
open the active CSV in a spreadsheet and check any answer by hand.

### Questions it handles

Aggregates and averages · counts · grouped statistics · extremes ("which region
is highest") · grouped filters ("which regions have more than 5 transactions")
· date ranges · numeric comparisons · missing-value questions.

These are examples, not a list of supported sentences.

---

## The data contract

Required columns: `id`, `date` (ISO `YYYY-MM-DD` only), `region`, `product`,
`units`, `unit_price`, `discount` (a **fraction** — `0.20` means 20 %). A file
missing any of them is rejected with a message naming what is missing.

```
revenue = units * unit_price * (1 - discount)
```

Extra columns are loaded and reported, but not analysable. Missing values are
skipped and **counted**, never zero-filled — an answer says *"computed from 9 of
the 10 matching transactions"* rather than quietly averaging over nine while
claiming ten.

---

## Guardrails

| Situation | Behaviour |
|---|---|
| Invented column or metric (`profit`) | Rejected, naming what is supported |
| Unknown category (`Britain`) | Rejected with the real values — synonyms are never mapped |
| A real value the question never named (`Germany` → `DE`) | Clarification requested, not assumed |
| Near miss (`Gama`) | "Did you mean 'Gamma'?" |
| Ambiguous wording ("in Mars") | Clarification requested, not guessed |
| Code execution, file access, secrets | Rejected before the model is called |
| A jailbroken model returning `{"tool": "run_shell"}` | No such tool exists — the plan fails validation and nothing runs |
| Dates outside the data | A valid question: answered "no data", with the real range |

Safety does not rest on the pre-screen or on the model behaving. It rests on
there being **no expressible operation outside six pure functions**. A test
asserts the `app/` package contains no `eval`, `exec`, `compile`,
`__import__`, `subprocess`, `os.system`, `pickle` or `shutil`, and that only
the data loader opens files.

---

## Configuration

Everything lives in `.env` (see `.env.example`). The API key is read by the
server only — it is never sent to the browser.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `gemini` |
| `GROQ_API_KEY` | – | required when the provider is `groq` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | planning model |
| `LLM_FALLBACK_PROVIDER` | `gemini` | asked only when the primary cannot answer at all |
| `GEMINI_API_KEY` | – | required for the fallback to be active |
| `GEMINI_MODEL` | `gemini-3.6-flash` | planning model |
| `GEMINI_API_KEY_2` | – | optional second key, tried on a 429 |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash-lite` | tried on a 503 |
| `GEMINI_THINKING_LEVEL` | `MINIMAL` | planning is translation, not reasoning |

---

## Deployment

One Render web service: FastAPI serves `/api/*` and, from the same origin, the
React production build. `render.yaml` holds the whole configuration — build
command, start command, health check and environment. The two API keys are
marked `sync: false`, so Render prompts for them and they never enter the
repository.

Being a free single-instance demo, it **sleeps after ~15 minutes idle** (the
first request back pays a 30–60 s cold start, which the UI explains rather than
spinning silently), **sessions do not survive a restart**, and **uploads are not
persisted**. Details and the reasoning are in
[ARCHITECTURE.md](ARCHITECTURE.md#5-deployment-shape).

---

## Limitations

* Plan quality depends on the model. The architecture guarantees a bad plan is
  *rejected* — not that every question produces one.
* Answers are single values, grouped values or an extreme. The agent does not
  list matching transactions.
* Free-tier quotas apply to both providers. When both are exhausted, questions
  return an honest `Status: Error` rather than a fabricated answer.
