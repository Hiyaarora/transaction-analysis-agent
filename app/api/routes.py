"""Five endpoints. Each one looks up a session, calls an existing function,
and converts the result. No analysis happens here.

    GET  /api/health
    POST /api/dataset/assessment    load the bundled dataset
    POST /api/dataset/upload        load an uploaded CSV
    GET  /api/dataset               the active dataset for this session
    GET  /api/dataset/download      the CSV itself, to open elsewhere
    POST /api/ask                   ask a question

Handlers are `def`, not `async def`, so FastAPI runs them in its threadpool:
planning blocks on a network call, and one slow question must not stall the
server.
"""

import os
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, Response, UploadFile

from app.agent import Agent
from app.api import convert, schemas
from app.api.deps import get_llm, get_store, llm_is_configured
from app.api.sessions import Session, SessionStore
from app.data_loader import ActiveDataset, DatasetLoadError, load_dataset
from app.llm.base import LLMClient

router = APIRouter(prefix="/api")

# The one path the server will ever open on its own initiative. The client
# cannot name a path: it either asks for this file or uploads bytes.
ASSESSMENT_DATASET = Path("data/project_4.csv")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
_TEMP_DIR: Path | None = None  # tests point this at tmp_path


def _fail(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _session_id(x_session_id: str = Header(min_length=8, max_length=64)) -> str:
    return x_session_id


def _active(session: Session | None) -> Agent:
    if session is None or session.agent is None:
        raise _fail(409, "no_dataset", "No dataset is loaded for this session. Load one first.")
    return session.agent


def _loaded(session: Session | None) -> Session:
    _active(session)  # same error when nothing is loaded
    return session


# --- health -------------------------------------------------------------------------


@router.get("/health", response_model=schemas.Health)
def health() -> schemas.Health:
    return schemas.Health(status="ok", llm_configured=llm_is_configured())


# --- dataset ------------------------------------------------------------------------


@router.get("/dataset", response_model=schemas.DatasetState)
def current_dataset(
    session_id: str = Depends(_session_id),
    store: SessionStore = Depends(get_store),
) -> schemas.DatasetState:
    return convert.dataset_state(_active(store.get(session_id)).dataset)


@router.get("/dataset/download")
def download_dataset(
    session_id: str = Depends(_session_id),
    store: SessionStore = Depends(get_store),
) -> Response:
    session = _loaded(store.get(session_id))
    # The name is already stripped of any path (see _safe_name), and is quoted
    # so it cannot break out of the header.
    return Response(
        content=session.source_bytes or b"",
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{session.source_name}"'},
    )


@router.post("/dataset/assessment", response_model=schemas.DatasetState)
def load_assessment_dataset(
    session_id: str = Depends(_session_id),
    store: SessionStore = Depends(get_store),
    llm: LLMClient = Depends(get_llm),
) -> schemas.DatasetState:
    try:
        dataset = load_dataset(ASSESSMENT_DATASET)
    except DatasetLoadError as exc:
        raise _fail(500, "assessment_unavailable", f"The bundled dataset could not be loaded: {exc}") from exc
    return _activate(store.get_or_create(session_id), dataset, llm,
                     source=ASSESSMENT_DATASET.read_bytes())


@router.post("/dataset/upload", response_model=schemas.DatasetState)
def upload_dataset(
    file: UploadFile = File(),
    session_id: str = Depends(_session_id),
    store: SessionStore = Depends(get_store),
    llm: LLMClient = Depends(get_llm),
) -> schemas.DatasetState:
    content = _validated_upload(file)
    session = store.get_or_create(session_id)

    # The browser supplies bytes, never a path. The server chooses the
    # location, and removes the file as soon as the loader has read it -
    # whether that succeeded or failed. Nothing re-reads it afterwards.
    temp_path = _write_temp(content)
    try:
        dataset = load_dataset(temp_path)
    except DatasetLoadError as exc:
        raise _fail(400, "invalid_dataset", str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    # Only replace the active dataset once loading has succeeded; a failed
    # upload leaves the session exactly as it was.
    return _activate(session, _renamed(dataset, _safe_name(file.filename)), llm, source=content)


# --- questions ----------------------------------------------------------------------


@router.post("/ask", response_model=schemas.AskResponse)
def ask(
    request: schemas.AskRequest,
    session_id: str = Depends(_session_id),
    store: SessionStore = Depends(get_store),
) -> schemas.AskResponse:
    question = request.question.strip()
    if not question:
        raise _fail(422, "empty_question", "The question is empty.")
    agent = _active(store.get(session_id))
    # A provider failure is an answer status, not an HTTP error: the agent
    # reports it honestly and the session stays usable.
    return convert.ask_response(agent.ask(question))


# --- helpers ------------------------------------------------------------------------


def _activate(
    session: Session, dataset: ActiveDataset, llm: LLMClient, source: bytes
) -> schemas.DatasetState:
    if session.agent is None:
        session.agent = Agent(dataset, llm)
    else:
        session.agent.load(dataset)  # the same call the CLI's /load makes
    # Replacing the dataset replaces the file behind it, so a download always
    # matches the dataset the answers came from.
    session.source_bytes = source
    session.source_name = dataset.source_name
    return convert.dataset_state(dataset)


def _renamed(dataset: ActiveDataset, name: str) -> ActiveDataset:
    """Carry the user's filename, not the server's temporary one.

    The loader names the dataset after the file it read, which for an upload
    is a throwaway temp file. Both the dataset and its profile are relabelled
    so nothing downstream - including the CLI renderer - ever shows it.
    """
    return replace(dataset, source_name=name, profile=replace(dataset.profile, source_name=name))


def _validated_upload(file: UploadFile) -> bytes:
    name = _safe_name(file.filename)
    if not name.lower().endswith(".csv"):
        raise _fail(400, "not_a_csv", "Only .csv files can be loaded.")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise _fail(400, "too_large", f"The file is too large; the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not content.strip():
        raise _fail(400, "empty_file", "The file is empty.")
    return content


def _write_temp(content: bytes) -> Path:
    handle, name = tempfile.mkstemp(suffix=".csv", dir=_TEMP_DIR)
    os.close(handle)
    path = Path(name)
    path.write_bytes(content)
    return path


def _safe_name(filename: str | None) -> str:
    """The client's filename is a label only: strip any path it may carry."""
    return Path(filename or "uploaded.csv").name
