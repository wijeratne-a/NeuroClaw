"""EU AI Act guardrails: allowlist, blocklist, exit 64 (EX_USAGE)."""

from __future__ import annotations

import re
import sys
from typing import Final

EX_USAGE: Final[int] = 64

TRANSPARENCY_NOTICE = (
    "This system utilizes emotion recognition for commercial content optimization. "
    "Disclosure required under EU Regulation (EU) 2024/1689 Article 50 where applicable."
)

COMPLIANCE_WARNING = (
    "NOTICE: This system utilizes an Emotion AI framework strictly for commercial content "
    "optimization. Use for personnel monitoring or educational assessment is strictly "
    "prohibited under EU Regulation (EU) 2024/1689."
)

MINDFUL_MARKETING_WARNING = (
    "Mindful Marketing: Prohibited use case detected. NeuroClaw is only for "
    "commercial content optimization."
)

# Canonical tag after normalization
CANONICAL_USE_CASE = "commercial_content_optimization"

# Allowlist: map these inputs to canonical commercial_content_optimization
ALLOWED_ALIASES: frozenset[str] = frozenset(
    {
        "commercial-content-optimization",
        "commercial-optimization",
        "marketing-optimization",
        "creative-testing",
        "brand-engagement-analysis",
    }
)


def _canon_key(s: str) -> str:
    """Lowercase; spaces and underscores become hyphens; collapse repeats."""
    t = re.sub(r"[\s_]+", "-", s.strip().lower())
    t = re.sub(r"-+", "-", t).strip("-")
    return t

# Case-insensitive fuzzy blocklist (Article 5(1)(f) style contexts)
_BLOCKED_PATTERN = re.compile(
    r"surveillance|workplace|education|employee[\s_-]*monitoring|"
    r"recruitment[\s_-]*screening|burnout[\s_-]*detection|"
    r"classroom[\s_-]*analysis|proctoring|student[\s_-]*attention[\s_-]*tracking|"
    r"personnel[\s_-]*assessment|interview[\s_-]*analytics|"
    r"biometric[\s_-]*profiling|emotion[\s_-]*identification|"
    r"wellbeing[\s_-]*monitoring|mental[\s_-]*health[\s_-]*screening",
    re.IGNORECASE | re.VERBOSE,
)


def normalize_use_case(raw: str) -> str:
    """Map allowlist aliases to canonical tag."""
    key = _canon_key(raw)
    if key in ALLOWED_ALIASES:
        return CANONICAL_USE_CASE
    return key


def validate_use_case(use_case: str) -> str:
    """
    Block prohibited contexts (regex), then require allowlist match.
    Exit with code EX_USAGE (64) on failure.
    Returns normalized canonical use_case string on success.
    """
    if not use_case or not str(use_case).strip():
        _exit_blocked("empty --use_case")
    raw = use_case.strip()
    if _BLOCKED_PATTERN.search(raw):
        _exit_blocked(f"blocked term in use_case: {raw!r}")
    normalized = normalize_use_case(raw)
    if normalized != CANONICAL_USE_CASE:
        _exit_blocked(
            f"use_case not allowlisted: {raw!r} "
            f"(allowed: {sorted(ALLOWED_ALIASES)})"
        )
    return CANONICAL_USE_CASE


def _exit_blocked(reason: str) -> None:
    """Log Mindful Marketing warning and exit 64."""
    print(COMPLIANCE_WARNING, file=sys.stderr)
    print(MINDFUL_MARKETING_WARNING, file=sys.stderr)
    print(f"Compliance failure: {reason}", file=sys.stderr)
    sys.exit(EX_USAGE)
