# Transaction Analysis Agent

A full-stack **agentic data-analysis application** that converts natural-language questions into a validated analysis plan and executes that plan with deterministic Python/Pandas operations.

> **Core boundary:** the LLM understands the question and chooses operations; Python validates, executes, and computes the numbers.

### Live Demo

**Live URL:** **https://transaction-analysis-agent.onrender.com**

Health check: [`/api/health`](https://transaction-analysis-agent.onrender.com/api/health)

## What it demonstrates

- Natural-language → structured `AnalysisPlan`
- Six restricted deterministic tools: `filter_rows`, `compute_metric`, `aggregate`, `group_by`, `filter_groups`, `select_extreme`
- Runtime discovery of dataset categories, dates, row counts, and missing values
- Pydantic schema validation + semantic plan validation
- Ambiguity → clarification; unsupported requests → rejection
- No arbitrary Python, shell, filesystem, or secret-access capability
- **Groq (`openai/gpt-oss-120b`) as default**, with Gemini as an automatic fallback
- React + TypeScript UI with assessment questions and custom questions
- CSV upload and dataset replacement
- Cached answered questions
- CLI + web UI using the same backend pipeline
- 537 automated tests (475 backend, 62 frontend)

📄 **[ARCHITECTURE.md](ARCHITECTURE.md)** — the full project document: architecture, end-to-end flow, component roles, the restricted tool layer, guardrails, worked examples from the supplied assessment, and the key design choices.

---

## Architecture

```text
React + TypeScript UI / CLI
             |
             v
        FastAPI API
             |
             v
          Agent
             |
          Planner
             |
       LLMClient
       /       \
    Groq      Gemini
             |
        AnalysisPlan
             |
        schemas.py
             |
        validator.py
             |
         executor.py
             |
           tools.py
             |
        Pandas/DataFrame
             |
        ExecutionResult
             |
        renderer.py
             |
        User-facing result
```

### End-to-end flow

1. The user asks a question.
2. A pre-screen stops obvious code/file/secret requests before any API call is made.
3. The planner receives the question plus dataset metadata—not transaction rows or file paths.
4. The LLM returns a structured `AnalysisPlan`.
5. `schemas.py` checks the plan structure.
6. `validator.py` checks whether the plan is valid for the active dataset and workflow.
7. `executor.py` runs only the approved tools.
8. Pandas computes the actual result.
9. `renderer.py` presents the result and recorded operations.

---

## Project structure

```text
transaction-analysis-agent/
├── app/
│   ├── contract.py       Domain vocabulary and derived metrics
│   ├── config.py         Environment/configuration
│   ├── data_loader.py    CSV → ActiveDataset
│   ├── data_profile.py   Runtime dataset profile
│   ├── schemas.py        AnalysisPlan models
│   ├── validator.py      Semantic validation
│   ├── tools.py          Deterministic Pandas tools
│   ├── executor.py       Plan execution
│   ├── renderer.py       Result → readable output
│   ├── planner.py        Question → AnalysisPlan
│   ├── prescreen.py      Early cost-saving filter
│   ├── agent.py          Pipeline orchestration
│   ├── cli.py            Interactive terminal session
│   ├── llm/
│   │   ├── base.py       LLMClient interface
│   │   ├── fake.py       Test double
│   │   ├── groq.py       Groq provider
│   │   ├── gemini.py     Gemini provider
│   │   ├── chain.py      Cross-provider fallback
│   │   └── factory.py    Provider selection
│   └── api/              FastAPI routes, wire schemas, sessions, static serving
├── frontend/             React + TypeScript UI
├── data/
│   └── project_4.csv
├── tests/
├── run.py                CLI entry point
├── run_api.py            FastAPI entry point
├── render.yaml           Render deployment config
├── .python-version       Pinned interpreter
└── ARCHITECTURE.md       Design document
```

---

## Deterministic tool layer

The LLM can only express operations through six approved tools:

| Tool | Purpose |
|---|---|
| `filter_rows` | Filter transactions using validated columns/operators |
| `compute_metric` | Compute derived metrics such as revenue |
| `aggregate` | Calculate `sum`, `mean`, `median`, `min`, `max`, or `count` |
| `group_by` | Group transactions and calculate an aggregate per group |
| `filter_groups` | Filter grouped results using validated comparisons |
| `select_extreme` | Select the highest/lowest group or value |

No tool accepts arbitrary Python code, shell commands, file paths, or expressions outside the allowed operation set.

---

## Data-driven behavior

The application does **not** hardcode transaction values, category names, dates, row counts, or answers.

The active CSV is profiled at runtime. A different valid CSV can be loaded with different data values and categories without changing the analysis code.

The UI supports:

- **Assessment dataset** — use the bundled `project_4.csv`
- **Custom CSV upload** — replace the active dataset
- **Assessment questions** — shown dynamically when a question set is available
- **Custom questions** — any supported natural-language analysis question
- **Download** — the active CSV can be downloaded to verify any answer by hand

A CSV **with** a `question` column has those rows set aside as the question list and excluded from every calculation. A CSV **without** one loads normally with an empty question list — ask your own questions instead.

---

## Example interactions

```text
"What is the total discounted revenue for Gamma?"
→ Planned → validated → executed → result returned

"Read a secret value from the machine."
→ Rejected
```

The pre-screen is only an early cost-saving filter. The real safety boundary is the **closed plan schema + semantic validator + restricted executor/tools**.

---

## Running locally

### 1. Setup

Requirements:
- Python 3.12
- Node.js/npm for the web UI
- A Groq API key

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

> On Windows, if PowerShell blocks `activate` with an execution-policy error, call the interpreter directly instead: `.venv\Scripts\python.exe run.py ...`

Install backend dependencies and configure the environment:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Then add your key to `.env`:

```text
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=<your-key>
```

Never commit API keys.

### 2. Run the CLI

```bash
python run.py --data data/project_4.csv
```

Useful CLI commands:

```text
/questions                  list available assessment questions
/ask 3                      ask question 3
3                           the same thing, typed faster
/profile                    show the active dataset profile
/load path/to/another.csv   replace the active dataset
/exit                       close the session
```

Anything that is not a command is treated as a question. While the model plans, the terminal shows `Thinking / 4s` on one line, erased before the answer appears.

### 3. Run the web UI locally

Start the API in one terminal:

```bash
python run_api.py
```

The API runs at:

```text
http://127.0.0.1:8000
```

Start the React/Vite frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

### 4. Run like the deployed app

Build the frontend and let FastAPI serve the production build from the same service:

```bash
cd frontend
npm install
npm run build
cd ..
python run_api.py
```

Then open `http://localhost:8000`.

---

## Testing

Backend tests:

```bash
pytest -W error
```

Frontend tests:

```bash
cd frontend
npx vitest run
```

Tests cover the loader/profile, tools, schemas, validator, planner/provider abstraction, execution, renderer, CLI, API, static serving, ambiguity/rejection, and dataset replacement.

Almost every test runs with no network and no API key: a `FakeLLMClient` returns scripted plans and records what it was asked, so a failing test means Python misbehaved rather than that a model phrased something differently. No expected answer is a literal from the supplied file — end-to-end tests compute their expectation with plain pandas over whatever CSV is in `data/`.

---

## Data contract

Required columns: `id`, `date` (ISO `YYYY-MM-DD`), `region`, `product`, `units`, `unit_price`, `discount` (`0.20` means 20%).

Derived revenue:

```text
revenue = units * unit_price * (1 - discount)
```

Extra columns are loaded and reported but are not automatically analysable. Missing values are skipped and counted rather than silently zero-filled.

---

## Guardrails

| Situation | Behaviour |
|---|---|
| Invented column or metric (`profit`) | Rejected |
| Unknown category (`Britain`) | Rejected with the real valid values |
| A real value the question never named (`Germany` planned as `DE`) | Clarification requested, not assumed |
| Near miss (`Gama`) | Clarification requested |
| Ambiguous wording | Clarification requested, not guessed |
| Code execution, file access, secrets | Rejected |
| Invalid model-generated tool/step | Fails schema/semantic validation; nothing executes |
| Dates outside the active dataset range | Valid plan; returns no data and reports the actual range |

Safety does not rely on the model behaving correctly. The executable surface is limited to the validated plan and six deterministic tools.

---

## Configuration

Everything lives in `.env` (see `.env.example`). The API key is read by the server only and is never sent to the browser.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `gemini` |
| `GROQ_API_KEY` | – | Required when using Groq |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Planning model |
| `LLM_FALLBACK_PROVIDER` | `gemini` | Asked only when the primary provider cannot answer at all |
| `GEMINI_API_KEY` | – | Required for the fallback (or when using Gemini directly) |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Planning model |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash-lite` | Tried when the primary model is saturated |
| `GEMINI_THINKING_LEVEL` | `MINIMAL` | Planning is translation, not reasoning |

**Why Groq is the default:** measured on the ten questions the dataset carries, with every answer checked against pandas — Groq answered 10/10 at a 1.2 s median; Gemini answered 10/10 at 30.2 s. Validation and execution together take under 10 ms, so effectively all latency is the provider's.

---

## Deployment

The app is designed as **one Render Web Service** so the evaluator receives a single URL:

```text
Browser
  ↓
Render
  ├── React production build
  └── /api/* → FastAPI → Agent → LLM provider
```

The API router is registered before the static handler, so no static route can shadow an endpoint; an unknown path falls back to `index.html` for the SPA, except under `/api`, where a missing endpoint stays a JSON 404. Serving the UI from the same origin as the API is also why there is no CORS configuration in production.

Uploaded CSVs are temporary and loaded into memory — the server names its own temp file and deletes it as soon as the loader has read it, after a failure as well as a success. Session state is in memory, so a service restart starts a fresh session.

---

## Tech stack

**Frontend:** React, TypeScript, Vite, Tailwind CSS  
**Backend:** Python, FastAPI, Uvicorn, Pandas, Pydantic  
**LLM:** Groq (`openai/gpt-oss-120b`), Gemini fallback  
**Testing:** pytest, Vitest, FakeLLMClient  
**Deployment:** Render
