"""Split notation sections while preserving nested groups and quoted values."""

from .errors import PeptideError


def split_top_level(text: str, delimiter: str, *, error_code: str) -> list[str]:
    pairs = {"[": "]", "{": "}", "(": ")"}
    stack = []
    quote = False
    escaped = False
    start = 0
    parts = []
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if quote:
            if character == "\\":
                escaped = True
            elif character == '"':
                quote = False
            continue
        if character == '"':
            quote = True
        elif character in pairs:
            stack.append(pairs[character])
        elif character in pairs.values():
            if not stack or character != stack.pop():
                raise PeptideError(error_code, f"Unmatched delimiter at position {index}")
        elif character == delimiter and not stack:
            parts.append(text[start:index])
            start = index + 1
    if stack or quote:
        raise PeptideError(error_code, "Unterminated group or quoted value")
    parts.append(text[start:])
    return parts
