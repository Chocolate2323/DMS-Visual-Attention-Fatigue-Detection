from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonArrayWriter:
    """逐帧写入 JSON 数组，避免把所有结果先攒在内存里。"""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._file = None
        self._first = True

    def __enter__(self) -> "JsonArrayWriter":
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", encoding="utf-8")
            self._file.write("[\n")
        return self

    def write(self, record: dict[str, Any]) -> None:
        if not self._file:
            return
        if not self._first:
            self._file.write(",\n")
        json.dump(record, self._file, ensure_ascii=False)
        self._first = False

    def __exit__(self, *args: object) -> None:
        if self._file:
            self._file.write("\n]\n")
            self._file.close()
            self._file = None
