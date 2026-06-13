"""Plotly figure builders for the live chart and the replay scrubber."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .state import state
from .utils import ema, format_config

def _build_chart(events, title=None):
    """Single chart builder used by both Live and Replay."""
    if not events:
        fig = go.Figure()
        fig.update_layout(title=title or "No data", height=560,
                          margin=dict(l=40, r=20, t=60, b=40))
        return fig

    train_evts = [e for e in events if e["type"] == "train_step"]
    eval_evts  = [e for e in events if e["type"] == "eval"]
    swap_evts  = [e for e in events if e["type"] in ("optimizer_swap", "lr_change")]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Loss", "Accuracy"))

    if train_evts:
        steps  = [e["step"] for e in train_evts]
        losses = [e["data"]["loss"] for e in train_evts]
        accs   = [e["data"]["acc"]  for e in train_evts]
        fig.add_trace(go.Scatter(x=steps, y=losses, line=dict(color="rgba(31,119,180,0.2)", width=1),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=ema(losses), name="train loss",
                                 line=dict(color="rgb(31,119,180)", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=ema(accs), name="train acc",
                                 line=dict(color="rgb(44,160,44)", width=2)), row=2, col=1)

    if eval_evts:
        es = [e["step"] for e in eval_evts]
        tl = [e["data"]["test_loss"] for e in eval_evts]
        ta = [e["data"]["test_acc"]  for e in eval_evts]
        fig.add_trace(go.Scatter(x=es, y=tl, name="test loss", mode="markers+lines",
                                 marker=dict(size=10, color="red"),
                                 line=dict(width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=es, y=ta, name="test acc", mode="markers+lines",
                                 marker=dict(size=10, color="orange"),
                                 line=dict(width=2, dash="dash")), row=2, col=1)

    for e in swap_evts:
        if e["type"] == "optimizer_swap":
            label, color = f"-> {e['data']['to']['name']}", "purple"
        else:
            label, color = f"lr={e['data']['to']:.2e}", "gray"
        fig.add_vline(x=e["step"], line_dash="dot", line_color=color, opacity=0.6,
                      annotation_text=label, annotation_position="top",
                      annotation_font=dict(size=10, color=color), row=1, col=1)

    fig.update_layout(height=560, hovermode="x unified", title=title,
                      margin=dict(l=40, r=20, t=60, b=40),
                      legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(title_text="step", row=2, col=1)
    return fig


def build_figure():
    if state["event_log"] is None:
        return _build_chart([], "No run yet - set things up on the left and hit Start")
    return _build_chart(state["event_log"].snapshot())


def build_figure_at(max_t):
    if state["event_log"] is None:
        return _build_chart([], "No experiment recorded yet"), "_No data._"
    events = [e for e in state["event_log"].snapshot() if e["t"] <= max_t]
    fig = _build_chart(events, title=f"Replay at t = {max_t:.1f}s")

    config = None
    for e in events:
        if e["type"] == "training_start":
            config = dict(e["data"]["config"])
        elif e["type"] == "optimizer_swap":
            config = dict(e["data"]["to"])
        elif e["type"] == "lr_change" and config:
            config["lr"] = e["data"]["to"]

    last_step = max((e["step"] for e in events if e["step"] is not None), default=0)
    status = f"**t = {max_t:.1f}s**  |  step {last_step}"
    if config:
        status += f"  |  active: `{format_config(config)}`"
    return fig, status


def get_replay_max_t():
    log = state["event_log"]
    if log is None: return 0.0
    events = log.snapshot()
    return events[-1]["t"] if events else 0.0
