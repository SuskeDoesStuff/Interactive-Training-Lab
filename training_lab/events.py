"""EventLog: append-only, thread-safe, timestamped record of a run. Single source
of truth that the chart, replay, and report all read from."""
import json
import threading
import time
from copy import deepcopy


class EventLog:
    """Append-only timestamped log. Thread-safe; serializable to JSON."""
    def __init__(self):
        self._lock = threading.Lock()
        self._events = []
        self._t0 = time.time()

    def log(self, event_type, data=None, step=None, epoch=None):
        event = {
            "t":     time.time() - self._t0,
            "type":  event_type,
            "step":  step,
            "epoch": epoch,
            "data":  deepcopy(data or {}),
        }
        with self._lock:
            self._events.append(event)
        return event

    def snapshot(self):
        with self._lock:
            return list(self._events)

    def filter(self, *types):
        return [e for e in self.snapshot() if e["type"] in types]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, indent=2, default=str)

    def __len__(self):
        with self._lock:
            return len(self._events)
