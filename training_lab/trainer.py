"""Optimizer factory and the pausable, hot-swappable Trainer with optional
ReduceLROnPlateau scheduling."""
import time

import torch
import torch.nn as nn

OPTIMIZERS = {
    "adam":    torch.optim.Adam,
    "adamw":   torch.optim.AdamW,
    "sgd":     torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}

def build_optimizer(model, name, lr, kwargs=None):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](model.parameters(), lr=lr, **(kwargs or {}))


class Trainer:
    def __init__(self, model, train_loader, test_loader, controls, event_log,
                 device, log_every=1):
        self.model        = model
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.controls     = controls
        self.event_log    = event_log
        self.device       = device
        self.criterion    = nn.CrossEntropyLoss()
        self.log_every    = log_every

        snap = controls.optimizer_snapshot()
        self.optimizer = build_optimizer(model, snap["name"], snap["lr"], snap["kwargs"])
        self._opt_state = snap
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        """ReduceLROnPlateau bound to the CURRENT optimizer, or None if disabled.
        Rebuilt on optimizer swap so it never points at a dead optimizer."""
        if not self.controls.use_scheduler:
            return None
        cfg = self.controls.scheduler_cfg
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=cfg.get("factor", 0.1),
            patience=int(cfg.get("patience", 10)),
            min_lr=cfg.get("min_lr", 1e-5),
        )

    def _log(self, event_type, data=None):
        return self.event_log.log(
            event_type, data,
            step=self.controls.step,
            epoch=self.controls.epoch,
        )

    def _sync_optimizer(self):
        """Apply any pending optimizer/lr changes from controls."""
        desired = self.controls.optimizer_snapshot()
        if desired["name"] != self._opt_state["name"] or desired["kwargs"] != self._opt_state["kwargs"]:
            old = self._opt_state
            self.optimizer = build_optimizer(self.model, desired["name"], desired["lr"], desired["kwargs"])
            self._opt_state = desired
            # Scheduler held a reference to the old optimizer; rebind it (fresh
            # plateau baseline, which is the right behaviour for a new optimizer).
            self.scheduler = self._build_scheduler()
            self._log("optimizer_swap", {"from": old, "to": desired})
        elif desired["lr"] != self._opt_state["lr"]:
            old_lr = self._opt_state["lr"]
            for g in self.optimizer.param_groups:
                g["lr"] = desired["lr"]
            self._opt_state["lr"] = desired["lr"]
            self._log("lr_change", {"from": old_lr, "to": desired["lr"], "source": "manual"})

    def _wait_if_paused(self):
        if self.controls.paused:
            self._log("paused")
            while self.controls.paused and not self.controls.stop_requested:
                time.sleep(0.1)
            if not self.controls.stop_requested:
                self._log("resumed")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in self.test_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            out = self.model(xb)
            total_loss += self.criterion(out, yb).item() * xb.size(0)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += xb.size(0)
        self.model.train()
        return total_loss / total, correct / total

    def _scheduler_step(self, test_loss):
        """Step the scheduler on the eval metric. If it changes the LR, push the
        new value back into controls/_opt_state so the UI and _sync_optimizer
        stay consistent, and log it as an lr_change (source=scheduler)."""
        if self.scheduler is None:
            return
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step(test_loss)
        new_lr = self.optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            self.controls.set_lr(new_lr)
            self._opt_state["lr"] = new_lr
            self._log("lr_change", {"from": old_lr, "to": new_lr, "source": "scheduler"})

    def run(self):
        self._log("training_start", {"config": self.controls.optimizer_snapshot(),
                                     "max_epochs": self.controls.max_epochs,
                                     "scheduler": (self.controls.scheduler_cfg
                                                   if self.controls.use_scheduler else None)})
        try:
            while self.controls.epoch < self.controls.max_epochs:
                if self.controls.stop_requested:
                    break

                self._log("epoch_start")
                self.model.train()

                for xb, yb in self.train_loader:
                    self._wait_if_paused()
                    if self.controls.stop_requested:
                        break

                    self._sync_optimizer()

                    xb, yb = xb.to(self.device), yb.to(self.device)
                    self.optimizer.zero_grad()
                    out  = self.model(xb)
                    loss = self.criterion(out, yb)
                    loss.backward()
                    self.optimizer.step()

                    if self.controls.step % self.log_every == 0:
                        acc = (out.argmax(1) == yb).float().mean().item()
                        self._log("train_step", {
                            "loss": loss.item(),
                            "acc":  acc,
                            "lr":   self.optimizer.param_groups[0]["lr"],
                            "optimizer": self._opt_state["name"],
                        })
                    self.controls.step += 1

                if self.controls.stop_requested:
                    break

                test_loss, test_acc = self.evaluate()
                self._log("eval", {"test_loss": test_loss, "test_acc": test_acc})
                self._scheduler_step(test_loss)
                self._log("epoch_end")
                self.controls.epoch += 1

        finally:
            self._log("training_end", {
                "final_step":  self.controls.step,
                "final_epoch": self.controls.epoch,
                "stopped_early": self.controls.stop_requested,
            })
