"""Settings come from the environment (.env), never from code."""

from app.config import load_settings


def test_defaults(monkeypatch):
    for var in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = load_settings(dotenv_path=None)
    assert s.llm_provider == "gemini"
    assert s.gemini_api_key is None
    assert s.gemini_model == "gemini-3.6-flash"
    assert s.gemini_fallback_model is None


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "")
    s = load_settings(dotenv_path=None)
    assert (s.llm_provider, s.gemini_api_key, s.gemini_model) == ("fake", "abc", "gemini-3.5-flash")
    assert s.gemini_fallback_model is None  # empty string means "no fallback"


def test_dotenv_file_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text("GEMINI_MODEL=from-file\n")
    assert load_settings(dotenv_path=env).gemini_model == "from-file"


def test_backup_key_is_optional(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    assert load_settings(dotenv_path=None).gemini_api_key_2 is None


def test_backup_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_2", "second")
    assert load_settings(dotenv_path=None).gemini_api_key_2 == "second"
