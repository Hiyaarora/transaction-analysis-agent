"""One process serves the API and the built UI. These tests fix the boundary
between them.

The build is invented here rather than taken from `frontend/dist`, so the
tests say the same thing whether or not anyone has run `npm run build` in this
checkout - and so the fallback behaviour is asserted against a file whose
contents the test itself chose.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app

INDEX_HTML = "<!doctype html><title>the app</title>"


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    """A stand-in for the Vite build: an index and one hashed asset."""
    build = tmp_path / "dist"
    (build / "assets").mkdir(parents=True)
    (build / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (build / "assets" / "app-abc123.js").write_text("console.log('ui')", encoding="utf-8")
    (build / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    # A file next to the build that must never be reachable through it.
    (tmp_path / "secret.txt").write_text("GROQ_API_KEY=not-a-real-key", encoding="utf-8")
    return build


@pytest.fixture
def client(dist: Path) -> TestClient:
    with TestClient(create_app(dist)) as test_client:
        yield test_client


def test_the_root_serves_the_built_app(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == INDEX_HTML


def test_build_files_are_served_as_themselves(client):
    assert client.get("/assets/app-abc123.js").text == "console.log('ui')"
    assert client.get("/favicon.svg").status_code == 200


def test_an_unknown_path_falls_back_to_the_app_not_a_404(client):
    # A single-page app routes in the browser, so reloading on any path has to
    # return the same document.
    response = client.get("/questions/3")
    assert response.status_code == 200
    assert response.text == INDEX_HTML


def test_an_unknown_api_path_stays_a_json_404(client):
    # The one thing the fallback must not do: answer a missing endpoint with
    # HTML, which the client would then fail to parse and report as something
    # else entirely.
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"
    assert "html" not in response.headers["content-type"]


def test_the_api_still_answers_with_the_frontend_mounted(client):
    assert client.get("/api/health").json()["status"] == "ok"


@pytest.mark.parametrize(
    "path",
    ["/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt", "/assets/%2e%2e%2f%2e%2e%2fsecret.txt"],
)
def test_no_path_can_escape_the_build_directory(client, path):
    response = client.get(path)
    assert "GROQ_API_KEY" not in response.text
    # Whatever the router made of it, the only document on offer is the app.
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.text == INDEX_HTML


def test_without_a_build_the_app_serves_the_api_only(tmp_path):
    """A checkout that has never been built still runs: that is the dev and
    test case, and it must not depend on a directory that is gitignored."""
    with TestClient(create_app(tmp_path / "never-built")) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 404
