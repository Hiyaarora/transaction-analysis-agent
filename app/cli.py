"""Command-line session: load a dataset once, answer many questions about it.

    python run.py --data data/project_4.csv

The dataset path is a command-line argument and nothing else in the system
sees it: the loader turns it into an ActiveDataset, and from then on the
planner only ever receives the profile.

Streams and the LLM client are parameters so the whole loop can be exercised
in tests without a terminal or a network.
"""

import argparse
from pathlib import Path
from typing import TextIO

from app.agent import Agent
from app.config import load_settings
from app.data_loader import ActiveDataset, DatasetLoadError, load_dataset
from app.llm.base import LLMClient, LLMError
from app.llm.factory import build_llm
from app.renderer import render

PROMPT = "> "

_HELP = """Commands:
  /help              show this message
  /profile           describe the active dataset (columns, values, ranges, missing)
  /questions         list the questions embedded in the dataset file, if any
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

    agent = Agent(dataset, client)
    say(_banner(dataset))
    say("Ready. Ask a question, or /help for commands.")
    say()

    for line in stdin:
        question = line.strip()
        if not question:
            continue
        if question.startswith("/"):
            keep_going = _command(question, agent, say)
            say()
            if not keep_going:
                return 0
            continue
        say(render(agent.ask(question), agent.dataset.profile))
        say()
    return 0


# --- commands -----------------------------------------------------------------------


def _command(line: str, agent: Agent, say) -> bool:
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
    elif name == "/load":
        if not argument:
            say("Usage: /load <path to a .csv file>")
        else:
            _load(argument, agent, say)
    else:
        say(f"Unknown command '{name}'. Type /help for the list.")
    return True


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
        lines.append(f"  The file also carries {len(dataset.questions)} questions (/questions).")
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
