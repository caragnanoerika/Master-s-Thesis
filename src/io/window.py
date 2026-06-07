"""Helpers for SV-ADF window identification."""
from __future__ import annotations


def window_id(start: str, end: str) -> str:
    """Construct a filename-safe window identifier."""
    return f"{start}_{end}"


def parse_window_id(wid: str) -> tuple[str, str]:
    """Recover (start, end) from a window_id."""
    start, end = wid.split("_")
    return start, end
