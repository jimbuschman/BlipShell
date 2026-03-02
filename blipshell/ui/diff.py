"""Colored unified diff generation for terminal display.

Extracted from cli.py for reuse by tool result display and approval prompts.
"""

import difflib


def generate_colored_diff(
    old_lines: list[str],
    new_lines: list[str],
    filename: str,
    max_lines: int = 50,
) -> str:
    """Generate a colored unified diff string using ANSI codes.

    Returns empty string if no differences found.
    """
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
        lineterm="",
    ))
    if not diff:
        return ""

    colored = []
    for i, line in enumerate(diff):
        if i >= max_lines:
            colored.append(f"\x1b[2m  ... ({len(diff) - max_lines} more lines)\x1b[0m")
            break
        if line.startswith("---") or line.startswith("+++"):
            colored.append(f"\x1b[1m{line}\x1b[0m")
        elif line.startswith("@@"):
            colored.append(f"\x1b[36m{line}\x1b[0m")
        elif line.startswith("-"):
            colored.append(f"\x1b[31m{line}\x1b[0m")
        elif line.startswith("+"):
            colored.append(f"\x1b[32m{line}\x1b[0m")
        else:
            colored.append(line)
    return "\n".join(colored)
