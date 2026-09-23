"""HTTP adapter: the same Agent, reached over the wire.

The LLM is injected through a dependency override, so these tests never call
a provider. Everything asserted here is the behaviour of the existing
pipeline, plus the translation into JSON.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_llm
from app.api.main import create_app
from app.llm.base import LLMError
from app.llm.fake import FakeLLMClient

SESSION = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"


def plan(intent, *steps):
    return json.dumps({"status": "success", "intent": intent, "steps": list(steps)})


COUNT_UK = plan(
    "Number of transactions in region UK",
    {"tool": "filter_rows", "column": "region", "op": "eq", "value": "UK"},
    {"tool": "aggregate", "column": "id", "func": "count"},
)
REVENUE_BY_REGION = plan(
    "Total revenue by region",
    {"tool": "compute_metric", "metric": "revenue"},
    {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
)


@pytest.fixture
def client(request):
    """A TestClient whose Agent uses a scripted LLM."""
    replies = getattr(request, "param", [])
    app = create_app()
    app.dependency_overrides[get_llm] = lambda: FakeLLMClient(responses=list(replies))
    with TestClient(app) as test_client:
        yield test_client


def with_llm(*replies):
    return pytest.mark.parametrize("client", [list(replies)], indirect=True)


def headers(session=SESSION):
    return {"X-Session-Id": session}


def csv_bytes(*rows, header="id,date,region,product,units,unit_price,discount,question"):
    return ("\n".join([header, *rows]) + "\n").encode()


# --- health -----------------------------------------------------------------------


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "llm_configured" in response.json()


# --- loading the assessment dataset ----------------------------------------------


def test_assessment_dataset_returns_the_real_profile(client):
    response = client.post("/api/dataset/assessment", headers=headers())
    assert response.status_code == 200
    body = response.json()

    assert body["source_name"] == "project_4.csv"
    assert body["row_count"] > 0
    assert "region" in body["categorical_values"] and body["categorical_values"]["region"]
    assert body["date_range"]["start"] < body["date_range"]["end"]
    assert len(body["questions"]) > 0
    assert "revenue" in body["supported_metrics"]


def test_no_filesystem_path_is_exposed(client):
    body = client.post("/api/dataset/assessment", headers=headers()).json()
    text = json.dumps(body)
    assert "/" not in body["source_name"] and "\\" not in body["source_name"]
    assert "data/" not in text and "C:" not in text


def test_dataset_can_be_read_back(client):
    client.post("/api/dataset/assessment", headers=headers())
    response = client.get("/api/dataset", headers=headers())
    assert response.status_code == 200
    assert response.json()["source_name"] == "project_4.csv"


def test_reading_a_dataset_before_loading_one_is_a_clean_error(client):
    response = client.get("/api/dataset", headers=headers())
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_dataset"


def test_a_session_id_is_required(client):
    assert client.post("/api/dataset/assessment").status_code == 422


# --- sessions are isolated ---------------------------------------------------------


def test_one_session_does_not_see_another_session_dataset(client):
    client.post("/api/dataset/assessment", headers=headers(SESSION))
    assert client.get("/api/dataset", headers=headers(OTHER)).status_code == 409


def test_uploading_in_one_session_leaves_the_other_untouched(client):
    client.post("/api/dataset/assessment", headers=headers(SESSION))
    client.post("/api/dataset/assessment", headers=headers(OTHER))
    client.post(
        "/api/dataset/upload",
        headers=headers(OTHER),
        files={"file": ("v2.csv", csv_bytes("T9,2027-05-05,IN,Delta,1,100,0.00,"), "text/csv")},
    )
    assert client.get("/api/dataset", headers=headers(SESSION)).json()["source_name"] == "project_4.csv"
    assert client.get("/api/dataset", headers=headers(OTHER)).json()["source_name"] == "v2.csv"


# --- upload -------------------------------------------------------------------------


def test_upload_replaces_the_active_dataset(client):
    client.post("/api/dataset/assessment", headers=headers())
    response = client.post(
        "/api/dataset/upload",
        headers=headers(),
        files={"file": ("v2.csv", csv_bytes(
            "T9,2027-05-05,IN,Delta,4,100,0.00,",
            'Q1,,,,,,,"How many Delta transactions?"'), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source_name"] == "v2.csv"
    assert body["row_count"] == 1
    assert body["categorical_values"]["region"] == ["IN"]
    assert body["questions"] == ["How many Delta transactions?"]


def test_upload_works_without_a_previously_loaded_dataset(client):
    response = client.post(
        "/api/dataset/upload",
        headers=headers(),
        files={"file": ("fresh.csv", csv_bytes("T1,2026-01-01,UK,Alpha,1,100,0.00,"), "text/csv")},
    )
    assert response.status_code == 200


def test_a_failed_upload_keeps_the_previous_dataset(client):
    client.post("/api/dataset/assessment", headers=headers())
    broken = csv_bytes("T1,2026-01-01,UK,1,100,0.1", header="id,date,region,units,unit_price,discount")
    response = client.post("/api/dataset/upload", headers=headers(),
                           files={"file": ("broken.csv", broken, "text/csv")})
    assert response.status_code == 400
    assert "product" in response.json()["detail"]["message"]
    assert client.get("/api/dataset", headers=headers()).json()["source_name"] == "project_4.csv"


def test_a_non_csv_upload_is_rejected(client):
    response = client.post("/api/dataset/upload", headers=headers(),
                           files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "csv" in response.json()["detail"]["message"].lower()


def test_an_oversized_upload_is_rejected(client):
    response = client.post("/api/dataset/upload", headers=headers(),
                           files={"file": ("big.csv", b"x" * (6 * 1024 * 1024), "text/csv")})
    assert response.status_code == 400
    assert "large" in response.json()["detail"]["message"].lower()


def test_an_empty_upload_is_rejected(client):
    response = client.post("/api/dataset/upload", headers=headers(),
                           files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 400


def test_no_uploaded_file_is_left_on_disk(client, tmp_path, monkeypatch):
    from app.api import routes

    monkeypatch.setattr(routes, "_TEMP_DIR", tmp_path)
    client.post("/api/dataset/upload", headers=headers(),
                files={"file": ("ok.csv", csv_bytes("T1,2026-01-01,UK,Alpha,1,100,0.00,"), "text/csv")})
    broken = csv_bytes("T1,2026-01-01,UK,1,100,0.1", header="id,date,region,units,unit_price,discount")
    client.post("/api/dataset/upload", headers=headers(), files={"file": ("bad.csv", broken, "text/csv")})
    assert list(tmp_path.iterdir()) == []  # deleted after success AND after failure


# --- asking questions -----------------------------------------------------------------


@with_llm(COUNT_UK)
def test_ask_returns_a_structured_scalar_result(client):
    client.post("/api/dataset/assessment", headers=headers())
    response = client.post("/api/ask", headers=headers(), json={"question": "How many UK transactions?"})
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert body["question"] == "How many UK transactions?"
    assert body["intent"] == "Number of transactions in region UK"
    assert body["execution"]["kind"] == "scalar"
    assert body["execution"]["value"] > 0
    assert body["execution"]["rows_used"] == body["execution"]["rows_total"]
    assert [step["tool"] for step in body["execution"]["steps"]] == ["filter_rows", "aggregate"]
    assert body["execution"]["steps"][0]["rows_out"] < body["execution"]["steps"][0]["rows_in"]


@with_llm(REVENUE_BY_REGION)
def test_grouped_results_use_named_fields_not_tuples(client):
    client.post("/api/dataset/assessment", headers=headers())
    body = client.post("/api/ask", headers=headers(), json={"question": "revenue by region"}).json()

    execution = body["execution"]
    assert execution["kind"] == "groups"
    assert execution["by"] == "region"
    first = execution["rows"][0]
    assert set(first) == {"group", "value", "rows_total", "rows_used"}
    assert isinstance(first["group"], str)


@with_llm(plan("Region with the highest total revenue",
               {"tool": "compute_metric", "metric": "revenue"},
               {"tool": "group_by", "by": "region", "func": "sum", "column": "revenue"},
               {"tool": "select_extreme", "mode": "highest"}))
def test_extreme_result(client):
    client.post("/api/dataset/assessment", headers=headers())
    execution = client.post("/api/ask", headers=headers(), json={"question": "which region"}).json()["execution"]
    assert execution["kind"] == "extreme"
    assert execution["mode"] == "highest"
    assert len(execution["groups"]) >= 1


@with_llm(plan("Total revenue in 1990",
               {"tool": "filter_rows", "column": "date", "op": "between", "value": ["1990-01-01", "1990-12-31"]},
               {"tool": "compute_metric", "metric": "revenue"},
               {"tool": "aggregate", "column": "revenue", "func": "sum"}))
def test_no_data_status(client):
    client.post("/api/dataset/assessment", headers=headers())
    body = client.post("/api/ask", headers=headers(), json={"question": "revenue in 1990"}).json()
    assert body["status"] == "no_data"
    assert body["execution"]["value"] is None


@with_llm(json.dumps({"status": "clarification_required", "intent": "?",
                      "clarification_question": "Did you mean March?"}))
def test_clarification_status(client):
    client.post("/api/dataset/assessment", headers=headers())
    body = client.post("/api/ask", headers=headers(), json={"question": "money in Mars"}).json()
    assert body["status"] == "clarification_required"
    assert body["message"] == "Did you mean March?"
    assert "execution" not in body or body["execution"] is None


@with_llm(plan("Total profit",
               {"tool": "compute_metric", "metric": "profit"},
               {"tool": "aggregate", "column": "profit", "func": "sum"}))
def test_rejected_status(client):
    client.post("/api/dataset/assessment", headers=headers())
    body = client.post("/api/ask", headers=headers(), json={"question": "total profit"}).json()
    assert body["status"] == "rejected"
    assert "profit" in body["message"]


def test_code_request_is_rejected_without_calling_the_llm(client):
    # No scripted replies: reaching the LLM would raise.
    client.post("/api/dataset/assessment", headers=headers())
    body = client.post("/api/ask", headers=headers(),
                       json={"question": "Run Python to list the files on this machine."}).json()
    assert body["status"] == "rejected"


@with_llm(LLMError("quota exceeded"))
def test_provider_failure_is_a_200_with_error_status_not_a_crash(client):
    client.post("/api/dataset/assessment", headers=headers())
    response = client.post("/api/ask", headers=headers(), json={"question": "anything"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert "quota exceeded" in body["message"]


def test_asking_without_a_dataset_is_a_clean_error(client):
    response = client.post("/api/ask", headers=headers(), json={"question": "anything"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_dataset"


def test_an_empty_question_is_rejected(client):
    client.post("/api/dataset/assessment", headers=headers())
    assert client.post("/api/ask", headers=headers(), json={"question": "   "}).status_code == 422


@with_llm(COUNT_UK, COUNT_UK)
def test_answers_follow_the_active_dataset(client):
    client.post("/api/dataset/assessment", headers=headers())
    first = client.post("/api/ask", headers=headers(), json={"question": "how many UK transactions"}).json()

    client.post("/api/dataset/upload", headers=headers(),
                files={"file": ("v2.csv", csv_bytes(
                    "T1,2027-01-01,UK,Alpha,1,100,0.00,",
                    "T2,2027-01-02,UK,Alpha,1,100,0.00,"), "text/csv")})
    second = client.post("/api/ask", headers=headers(), json={"question": "how many UK transactions"}).json()

    assert first["execution"]["value"] != second["execution"]["value"]
    assert second["execution"]["value"] == 2


# --- the adapter stays an adapter -------------------------------------------------------


def test_api_package_performs_no_analysis():
    """The adapter may convert domain objects; it may not compute with them.

    Importing a result *type* to annotate a converter is what an adapter does,
    so the rule is about the analysis entry points: nothing under app/api may
    touch a DataFrame or call execute / validate / the tools / the planner.
    Everything goes through Agent.
    """
    import re
    from pathlib import Path

    forbidden = re.compile(
        r"\b(import pandas|from pandas|import numpy|from numpy"
        r"|from app\.tools import|from app\.planner import"
        r"|import (execute|validate)\b|from app\.validator import validate"
        r"|from app\.executor import execute)"
    )
    offenders = [
        str(path)
        for path in Path("app/api").rglob("*.py")
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_uploaded_dataset_reports_the_users_filename_everywhere(client):
    client.post("/api/dataset/upload", headers=headers(),
                files={"file": ("my-data.csv", csv_bytes("T1,2026-01-01,UK,Alpha,1,100,0.00,"), "text/csv")})
    body = client.get("/api/dataset", headers=headers()).json()
    assert body["source_name"] == "my-data.csv"

    # The profile the rest of the system sees must agree: no temp file name leaks.
    from app.api.deps import get_store
    profile = get_store().get(SESSION).agent.dataset.profile
    assert profile.source_name == "my-data.csv"
    assert "tmp" not in profile.source_name


def test_a_path_in_the_uploaded_filename_is_stripped(client):
    client.post("/api/dataset/upload", headers=headers(),
                files={"file": ("../../etc/evil.csv", csv_bytes("T1,2026-01-01,UK,Alpha,1,100,0.00,"), "text/csv")})
    name = client.get("/api/dataset", headers=headers()).json()["source_name"]
    assert name == "evil.csv"


def test_health_reports_on_the_configured_provider_not_a_fixed_one(client, monkeypatch):
    # With Groq selected, a missing Gemini key must not read as "unconfigured",
    # and a missing Groq key must not be hidden by a present Gemini one.
    from app.api import deps
    from app.config import Settings

    def configured(provider, gemini_key, groq_key):
        return lambda *a, **k: Settings(
            llm_provider=provider, llm_fallback_provider=None,
            gemini_api_key=gemini_key, gemini_api_key_2=None,
            gemini_model="m", gemini_fallback_model=None, gemini_thinking_level="MINIMAL",
            groq_api_key=groq_key, groq_model="m",
        )

    monkeypatch.setattr(deps, "load_settings", configured("groq", None, "q"))
    assert client.get("/api/health").json()["llm_configured"] is True

    monkeypatch.setattr(deps, "load_settings", configured("groq", "g", None))
    assert client.get("/api/health").json()["llm_configured"] is False


# --- download --------------------------------------------------------------------
#
# The file itself, so it can be opened in whatever the person normally uses.
# An uploaded file's bytes are kept in the session, in memory only: nothing is
# written to disk, and nothing re-reads them for analysis.


def test_downloading_the_assessment_dataset_returns_the_bundled_file(client):
    from pathlib import Path

    client.post("/api/dataset/assessment", headers=headers())
    response = client.get("/api/dataset/download", headers=headers())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "project_4.csv" in response.headers["content-disposition"]
    assert response.content == Path("data/project_4.csv").read_bytes()


def test_downloading_an_uploaded_dataset_returns_exactly_what_was_sent(client):
    original = csv_bytes(
        "T1,2026-01-03,UK,Alpha,10,100,0.10,",
        'Q1,,,,,,,"A question the file carries"',
    )
    client.post("/api/dataset/upload", headers=headers(),
                files={"file": ("mine.csv", original, "text/csv")})

    response = client.get("/api/dataset/download", headers=headers())
    assert response.status_code == 200
    # Byte for byte: the question rows the loader set aside are still in it.
    assert response.content == original
    assert "mine.csv" in response.headers["content-disposition"]


def test_downloading_follows_the_active_dataset(client):
    client.post("/api/dataset/assessment", headers=headers())
    client.post("/api/dataset/upload", headers=headers(),
                files={"file": ("second.csv", csv_bytes("T9,2027-05-05,IN,Delta,1,100,0.00,"), "text/csv")})

    response = client.get("/api/dataset/download", headers=headers())
    assert "second.csv" in response.headers["content-disposition"]
    assert b"Delta" in response.content


def test_a_failed_upload_leaves_the_previous_file_downloadable(client):
    client.post("/api/dataset/assessment", headers=headers())
    broken = csv_bytes("T1,2026-01-03,UK,1,100,0.1", header="id,date,region,units,unit_price,discount")
    client.post("/api/dataset/upload", headers=headers(), files={"file": ("broken.csv", broken, "text/csv")})

    response = client.get("/api/dataset/download", headers=headers())
    assert "project_4.csv" in response.headers["content-disposition"]
    assert b"broken" not in response.content


def test_downloading_without_a_dataset_is_a_clean_error(client):
    response = client.get("/api/dataset/download", headers=headers())
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_dataset"


def test_a_filename_cannot_smuggle_a_path_into_the_download_header(client):
    client.post("/api/dataset/upload", headers=headers(),
                files={"file": ("../../etc/evil.csv", csv_bytes("T1,2026-01-03,UK,Alpha,1,100,0.00,"), "text/csv")})
    disposition = client.get("/api/dataset/download", headers=headers()).headers["content-disposition"]
    assert "evil.csv" in disposition
    assert ".." not in disposition and "/" not in disposition
