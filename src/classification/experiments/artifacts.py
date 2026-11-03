"""Shared experiment artifact writing."""

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    """Write finite JSON through a sibling temporary file, then replace path.

    Output uses UTF-8 and ends with a newline. The caller must create the parent
    directory. Serialization and filesystem errors propagate; failed replacement
    can leave the temporary file behind.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
