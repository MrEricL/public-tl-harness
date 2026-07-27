from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class StageState:
    def __init__(self) -> None:
        self.status = "ok"
        self.payload: dict[str, object] = {}

    def finish(self, *, status: str = "ok", **payload: object) -> None:
        self.status = status
        self.payload.update(payload)


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **payload: object) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @contextmanager
    def stage(self, name: str, **payload: object) -> Iterator[StageState]:
        start = time.perf_counter()
        state = StageState()
        try:
            yield state
        except Exception as exc:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self.write(stage=name, status="fail", duration_ms=duration_ms, error=str(exc), **payload, **state.payload)
            raise
        else:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self.write(stage=name, status=state.status, duration_ms=duration_ms, **payload, **state.payload)
