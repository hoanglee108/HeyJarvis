"""Make Windows consoles safe for Vietnamese output.

The default Windows code page (cp1252/cp1258) cannot encode most Vietnamese
diacritics, so anything we print would raise ``UnicodeEncodeError``. Reconfiguring
the standard streams to UTF-8 (with a replacement fallback for legacy terminals)
fixes both ``print`` and logging.
"""

from __future__ import annotations

import sys


def enable_utf8_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic terminals
            pass
