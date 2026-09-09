"""Console encoding safety.

Windows consoles and redirected stdout are cp1252 by default; any model reply
that contains an emoji or a smart quote then raises UnicodeEncodeError from
`print`, and Rich inherits that. The 2026-09-09 gate run finished all three
scenarios and lost its JSON because the CLI crashed printing a reply's
checkmark BEFORE it reached the output write. Two rules follow: results are
written before anything cosmetic (see simulate_cmd), and the streams never
raise on an unencodable character - they substitute.
"""

from __future__ import annotations

import sys


def harden_stdio() -> None:
    """Make stdout/stderr substitute rather than raise on characters their
    encoding cannot represent. No-op for UTF-8 streams and for streams that
    cannot be reconfigured (pytest capture, some pipes)."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if stream is None or enc in ("utf8", "utf_8"):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
