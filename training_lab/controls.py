"""Controls: shared, mutable training config. The UI writes; the trainer reads."""
import threading


class Controls:
    """Shared, mutable training config. The UI writes; the trainer reads."""
    def __init__(self, optimizer="adam", lr=1e-3, max_epochs=10, optimizer_kwargs=None,
                 use_scheduler=False, scheduler_cfg=None):
        self._lock = threading.RLock()
        self.optimizer_name = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        # ReduceLROnPlateau config. Read once when the Trainer is built.
        self.use_scheduler = use_scheduler
        self.scheduler_cfg = dict(scheduler_cfg or {"factor": 0.1, "patience": 10, "min_lr": 1e-5})
        self.paused = False
        self.stop_requested = False
        self.step = 0
        self.epoch = 0

    def set_optimizer(self, name, lr=None, **kwargs):
        with self._lock:
            self.optimizer_name = name
            if lr is not None:
                self.lr = lr
            self.optimizer_kwargs = dict(kwargs)

    def set_lr(self, lr):
        with self._lock:
            self.lr = lr

    def set_max_epochs(self, n):
        with self._lock:
            self.max_epochs = int(n)

    def pause(self):  self.paused = True
    def resume(self): self.paused = False
    def stop(self):   self.stop_requested = True

    def optimizer_snapshot(self):
        with self._lock:
            return {
                "name": self.optimizer_name,
                "lr": float(self.lr),
                "kwargs": dict(self.optimizer_kwargs),
            }
