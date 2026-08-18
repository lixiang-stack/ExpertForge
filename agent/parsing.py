from __future__ import annotations

import json


def parse_json(text: str) -> dict | None:
    """Parse a pure-JSON response into a dict.

    Structured-output modes (json_schema/json_object) guarantee pure JSON, so
    there is no extraction fallback: unparseable output is treated as an error
    (``None``) and callers degrade via their existing validation paths. The
    greedy ``re.search(r"\\{.*\\}")`` approach is never used.
    """
    if not text:
        return None
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None