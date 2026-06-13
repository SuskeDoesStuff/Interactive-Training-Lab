"""Shared application state.

A single module-level dict that the UI mutates and the trainer/chart/report
code reads. Runtime dependencies (loaders, device) are injected once at startup
by app.main() so the handler functions can stay free of constructor plumbing.
"""

state = {
    # per-run objects, created on Start
    "spec": None,
    "model": None,
    "controls": None,
    "event_log": None,
    "trainer": None,
    "thread": None,
    # runtime deps injected at startup
    "train_loader": None,
    "test_loader": None,
    "device": None,
}


def is_running():
    t = state["thread"]
    return t is not None and t.is_alive()
