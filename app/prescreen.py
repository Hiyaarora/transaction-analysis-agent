"""A narrow, high-confidence pre-screen. Cost saving, not security.

WHAT IT DOES
    Stops requests that are unmistakably system-level — explicit code or
    command execution, explicit filesystem access, explicit secret extraction
    — before the planner is called. That saves an API call, ~1,300 prompt
    tokens, the latency, and a slot in the provider's daily quota.

WHAT IT DOES NOT DO
    It does not try to recognise every unsafe rephrasing, role-play framing
    ("act as a terminal"), or social-engineering wording. It is NOT the
    security boundary, and it must not grow into one: each new natural
    language pattern buys little and risks blocking a legitimate question.

WHY THAT IS SAFE
    A request this screen misses still meets, in order:
      1. the planner's refusal rules;
      2. the AnalysisPlan schema — only six tools are expressible, and
         `extra="forbid"` rejects any smuggled field;
      3. the semantic validator — unknown columns, metrics and values are
         rejected against the dataset profile;
      4. the executor — dispatch on step type to six pandas functions;
      5. the tools themselves, which re-check their own arguments.
    There is no code path to arbitrary execution, file access or the
    filesystem anywhere in the tool layer (enforced by a static test in
    tests/test_guardrails.py), so there is nothing for a cleverer sentence
    to unlock.

Every pattern below pairs an action with a concrete object, so ordinary
analytics wording — "discount code", "product line", "token size",
"list the regions" — cannot trip it.
"""

import re
from dataclasses import dataclass

_REASON = (
    "I can only analyse the loaded transaction dataset using the supported filter, compute, "
    "aggregate, group and compare operations. I cannot run code, execute commands, access files, "
    "or reveal secrets."
)

_PATTERNS = [
    # explicit code / command execution
    r"\b(run|execute|exec)\b.{0,30}\b(python|javascript|sql|code|script|shell|bash|powershell|command)\b",
    # code literals: these are not English, they are source
    r"\bimport\s+(os|sys|subprocess|shutil|socket|pathlib)\b|\b(subprocess|__import__)\b|\bos\.(system|popen)\b|(?<![.\w])(eval|exec)\s*\(",
    # a concrete filesystem path or sensitive file extension
    r"(^|[\s\"'(])(/etc/|/home/|/var/|/root/|~/|\.\./|[a-z]:\\)|\.(env|ssh|pem)\b",
    # file operations naming files/directories as the object
    r"\b(read|open|delete|remove|write|inspect|dump)\b.{0,20}\b(files?|directory|directories|folders?|file ?system)\b",
    # shell invocations with an argument
    r"\b(cat|ls|rm|chmod|curl|wget)\s+(-\w+\s+)?[~/.\\][\w./\\-]*",
    # secret extraction: a verb plus a named credential
    r"\b(show|reveal|print|give|tell|dump|extract|leak|expose)\b.{0,30}\b(api[ _-]?keys?|passwords?|credentials?|private keys?|secret keys?)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


@dataclass(frozen=True)
class PrescreenResult:
    blocked: bool
    reason: str | None = None


def prescreen(question: str) -> PrescreenResult:
    if any(p.search(question) for p in _COMPILED):
        return PrescreenResult(blocked=True, reason=_REASON)
    return PrescreenResult(blocked=False)
