from __future__ import annotations

import json
import logging
import statistics
import sys
import time
import uuid
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("sidi-insight")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


class LatencyTracker:
    """Rastreia latencias dos ultimos N requests para calcular P95."""

    def __init__(self, window: int = 20) -> None:
        self._lats: deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float) -> None:
        self._lats.append(latency_ms)

    def p95(self) -> float | None:
        if len(self._lats) < 2:
            return None
        sorted_lats = sorted(self._lats)
        idx = int(len(sorted_lats) * 0.95)
        return sorted_lats[idx]

    def avg(self) -> float | None:
        if not self._lats:
            return None
        return statistics.mean(self._lats)

    def count(self) -> int:
        return len(self._lats)


_latency_tracker = LatencyTracker()


def get_latency_tracker() -> LatencyTracker:
    return _latency_tracker


def log_event(event: str, trace_id: str | None = None, **fields: Any) -> None:
    payload = {
        "ts": time.time(),
        "event": event,
        "trace_id": trace_id or str(uuid.uuid4()),
        **fields,
    }
    logger.info(json.dumps(payload, default=str))


@contextmanager
def trace(operation: str, trace_id: str | None = None, **fields: Any) -> Iterator[dict[str, Any]]:
    tid = trace_id or str(uuid.uuid4())
    start = time.perf_counter()
    log_event(f"{operation}_start", trace_id=tid, **fields)
    ctx: dict[str, Any] = {"trace_id": tid}
    try:
        yield ctx
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        _latency_tracker.record(latency_ms)
        ctx["latency_ms"] = latency_ms
        merged = {**fields, **ctx}
        merged.pop("trace_id")
        log_event(f"{operation}_end", trace_id=tid, **merged)
