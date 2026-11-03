"""Shared experiment artifact writing."""

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    """Safely clean and write JSON config or results file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


