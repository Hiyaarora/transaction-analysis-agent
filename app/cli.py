"""Command-line session: load a dataset once, answer many questions about it.

    python run.py --data data/project_4.csv

The dataset path is a command-line argument and nothing else in the system
sees it: the loader turns it into an ActiveDataset, and from then on the
planner only ever receives the profile.

Streams and the LLM client are parameters so the whole loop can be exercised
in tests without a terminal or a network.
"""

import argparse
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from app.agent import Agent
from app.config import load_settings
from app.data_loader import ActiveDataset, DatasetLoadError, load_dataset
from app.llm.base import LLMClient, LLMError
from app.llm.factory import build_llm
from app.renderer import render

PROMPT = "> "


class Thinking:
    """A spinner for the wait, drawn only when a person is watching it.

    Planning is a network call - about a second on Groq, half a minute on
    Gemini - and a silent terminal for that long is indistinguishable from a
    hang. The frame changes so the wait looks alive, and the elapsed seconds
    say how long it has actually been.

    Two things make this safe to add to a program whose output is also a test
    fixture. It writes nothing at all unless the stream is a terminal, so a
    redirected or captured session gets exactly the bytes it got before. And
    it draws on one line with a carriage return, erasing itself on the way
    out, so the answer that follows starts on a clean line.

    The drawing runs on a daemon thread: the work it is reporting on is a
    blocking call, so nothing else would get a chance to draw. `tick` is
    separate from that thread and takes the step number, which is what lets a
    test assert what gets drawn without timing anything.
    """

    FRAMES = "|/-\\"
    LABEL = "Thinking"

    def __init__(
        self,
        stream: TextIO,
        interval: float = 0.12,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stream = stream
        self.interval = interval
        self._clock = clock
        # A stream that cannot say either way is assumed not to be a terminal.
        self.enabled = bool(getattr(stream, "isatty", lambda: False)())
        self._started = self._clock()
        self._width = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Thinking":
        self._started = self._clock()
        if self.enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exception: object) -> bool:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._erase()
        return False  # never swallow whatever went wrong inside the wait

    def tick(self, step: int) -> None:
        """Draw one frame. Does nothing when nobody is watching."""
        if not self.enabled:
            return
        elapsed = int(self._clock() - self._started)
        seconds = f" {elapsed}s" if elapsed else ""
        line = f"{self.LABEL} {self.FRAMES[step % len(self.FRAMES)]}{seconds}"
        self._width = max(self._width, len(line))
        self._write(f"\r{line}")

    def _spin(self) -> None:
        step = 0
        while not self._stop.is_set():
            self.tick(step)
            step += 1
            if self._stop.wait(self.interval):
                return

    def _erase(self) -> None:
        self._write("\r" + " " * self._width + "\r")

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except ValueError:  # the stream was closed while we were drawing
            self.enabled = False

_HELP = """Commands:
  /help              show this message
  /profile           describe the active dataset (columns, values, ranges, missing)
  /questions         list the questions embedded in the dataset file, if any
  /ask <n>           ask question n from that list - or just type the number
  /load <path>       load a different dataset for the rest of the session
  /exit              end the session
Anything else is treated as a question about the active dataset."""


def run(argv: list[str], stdin: TextIO, stdout: TextIO, llm: LLMClient | None = None) -> int:
    args = _parse(argv)

    def say(text: str = "") -> None:
        print(text, file=stdout)

    try:
        dataset = load_dataset(args.data)
    except DatasetLoadError as exc:
        say(f"Could not load the dataset: {exc}")
        return 1

    try:
        client = llm or _build_llm(use_dotenv=not args.no_dotenv)
    except LLMError as exc:
        say(f"Could not start the planner: {exc}")
        return 1

    def answer(question: str) -> str:
        """Ask, showing the wait, and render what comes back."""
        with Thinking(stdout):
            response = agent.ask(question)
        return render(response, agent.dataset.profile)

    agent = Agent(dataset, client)
    say(_banner(dataset))
    say("Ready. Ask a question, or /help for commands.")
    say()

    for line in stdin:
        question = line.strip()
        if not question:
            continue
        if question.startswith("/"):
            keep_going = _command(question, agent, say, answer)
            say()
            if not keep_going:
                return 0
            continue
        # A bare number names one of the file's own questions. Nothing else is
        # reinterpreted: "3" is not a question anybody could mean literally,
        # and sending it to the planner would spend a call to be told so.
        if _is_a_number(question):
            _ask_numbered(question, agent, say, answer)
            say()
            continue
        say(answer(question))
        say()
    return 0


# --- commands -----------------------------------------------------------------------


def _command(line: str, agent: Agent, say, answer) -> bool:
    """Handle a /command. Returns False when the session should end."""
    name, _, argument = line.partition(" ")
    name, argument = name.lower(), argument.strip()

    if name in ("/exit", "/quit"):
        say("Session ended.")
        return False
    if name == "/help":
        say(_HELP)
    elif name == "/profile":
        say(_profile_text(agent.dataset))
    elif name == "/questions":
        questions = agent.dataset.questions
        if questions:
            say(f"Questions embedded in {agent.dataset.source_name}:")
            for number, question in enumerate(questions, start=1):
                say(f"  {number}. {question}")
        else:
            say("The dataset file contains no embedded questions.")
    elif name == "/ask":
        _ask_numbered(argument, agent, say, answer)
    elif name == "/load":
        if not argument:
            say("Usage: /load <path to a .csv file>")
        else:
            _load(argument, agent, say)
    else:
        say(f"Unknown command '{name}'. Type /help for the list.")
    return True


def _ask_numbered(argument: str, agent: Agent, say, answer) -> None:
    """Ask the question the active file carries at position `argument`.

    The number is only a way of naming a question. Once resolved, the text
    goes through the same `agent.ask` a typed question does - same prescreen,
    same planner, same validation - so a question cannot reach the engine by a
    different route for having been chosen from a list.

    Nothing about the questions lives here: the list, its length and its
    wording all come from whatever CSV is loaded now.
    """
    questions = agent.dataset.questions
    if not questions:
        say("The dataset file contains no embedded questions.")
        return
    if not argument:
        say(f"Usage: /ask <number>, from 1 to {len(questions)}. Use /questions to see them.")
        return
    try:
        number = int(argument)
    except ValueError:
        say(f"'{argument}' is not a question number. Use /questions to see the list.")
        return
    # Explicitly 1-based on both ends: /ask 0 and /ask -1 would otherwise be
    # valid Python indices and quietly return the last question.
    if not 1 <= number <= len(questions):
        say(f"There is no question {number}. {agent.dataset.source_name} carries {len(questions)}.")
        return

    question = questions[number - 1]
    say(f"Q{number}: {question}")  # the transcript should say what was asked
    say(answer(question))


def _is_a_number(text: str) -> bool:
    """Whether the whole line is an integer - including a negative one, so that
    `/ask -1` and `-1` fail the same way rather than one of them being a
    question about minus one."""
    return text.removeprefix("-").isdigit()


def _load(path: str, agent: Agent, say) -> None:
    try:
        dataset = load_dataset(path)
    except DatasetLoadError as exc:
        say(f"The dataset could not be loaded: {exc}")
        say(f"Still using {agent.dataset.source_name}.")
        return
    agent.load(dataset)
    say(_banner(dataset))


# --- presentation -------------------------------------------------------------------


def _banner(dataset: ActiveDataset) -> str:
    profile = dataset.profile
    lines = [f"Loaded {dataset.source_name}: {profile.row_count} transactions."]
    lines.append(f"  Columns: {', '.join(profile.columns)}")
    if profile.date_range:
        start, end = profile.date_range
        lines.append(f"  Dates:   {start} to {end}")
    missing = [f"{column} ({count})" for column, count in profile.null_counts.items() if count]
    lines.append("  Missing: " + (", ".join(missing) if missing else "none"))
    if profile.parse_error_counts:
        unparsed = ", ".join(f"{column} ({count})" for column, count in profile.parse_error_counts.items())
        lines.append(f"  Unreadable values: {unparsed} (treated as missing)")
    if dataset.questions:
        lines.append(
            f"  The file also carries {len(dataset.questions)} questions "
            f"(/questions to list them, /ask 1 to run one)."
        )
    return "\n".join(lines)


def _profile_text(dataset: ActiveDataset) -> str:
    profile = dataset.profile
    lines = [f"Active dataset: {dataset.source_name} ({profile.row_count} transactions)"]
    for column, values in profile.categorical_values.items():
        lines.append(f"  {column}: {', '.join(values) if values else '(no values)'}")
    for column, span in profile.numeric_ranges.items():
        lines.append(f"  {column}: {span[0]:g} to {span[1]:g}" if span else f"  {column}: (no values)")
    if profile.date_range:
        start, end = profile.date_range
        lines.append(f"  date: {start} to {end}")
    lines.append(f"  Supported metrics: {', '.join(profile.supported_metrics)}")
    if profile.extra_columns:
        lines.append(f"  Present but not supported for analysis: {', '.join(profile.extra_columns)}")
    missing = [f"{column} ({count})" for column, count in profile.null_counts.items() if count]
    lines.append("  Missing values: " + (", ".join(missing) if missing else "none"))
    return "\n".join(lines)


# --- wiring -------------------------------------------------------------------------


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Ask natural-language questions about a transaction dataset."
    )
    parser.add_argument("--data", required=True, type=Path, help="path to the transaction CSV")
    parser.add_argument("--no-dotenv", action="store_true", help="ignore .env and read settings from the environment")
    return parser.parse_args(argv)


def _build_llm(use_dotenv: bool) -> LLMClient:
    settings = load_settings(dotenv_path=".env" if use_dotenv else None)
    return build_llm(settings)


def main() -> int:  # pragma: no cover - thin wrapper around run()
    import sys

    return run(sys.argv[1:], stdin=sys.stdin, stdout=sys.stdout)
