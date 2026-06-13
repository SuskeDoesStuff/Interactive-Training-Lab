"""Turn an EventLog into a human-readable report: slice the run into config
'eras', summarize each, and export Markdown / self-contained HTML / raw JSON."""
import json
import os

from .charts import build_figure
from .config import ensure_output_dir
from .state import state
from .utils import format_config


def analyze_experiment(event_log):
    events = event_log.snapshot()
    if not events: return []
    eras, current = [], None

    for e in events:
        if e["type"] == "training_start":
            current = {"config": dict(e["data"]["config"]), "start_step": 0,
                       "start_t": e["t"], "transition_in": None,
                       "train_steps": [], "evals": []}
        elif e["type"] in ("optimizer_swap", "lr_change") and current:
            current["end_step"] = e["step"]; current["end_t"] = e["t"]
            eras.append(current)
            new_cfg = dict(e["data"]["to"]) if e["type"] == "optimizer_swap" \
                      else {**current["config"], "lr": e["data"]["to"]}
            current = {"config": new_cfg, "start_step": e["step"], "start_t": e["t"],
                       "transition_in": e, "train_steps": [], "evals": []}
        elif e["type"] == "train_step" and current:
            current["train_steps"].append(e)
        elif e["type"] == "eval" and current:
            current["evals"].append(e)
        elif e["type"] == "training_end" and current:
            current["end_step"] = e["step"]; current["end_t"] = e["t"]
            eras.append(current); current = None

    if current is not None:
        current["end_step"] = events[-1].get("step") or current["start_step"]
        current["end_t"] = events[-1]["t"]
        eras.append(current)

    for era in eras:
        ts = era["train_steps"]
        if ts:
            losses = [t["data"]["loss"] for t in ts]
            era["mean_loss"]  = sum(losses) / len(losses)
            era["min_loss"]   = min(losses)
            era["start_loss"] = losses[0]
            era["end_loss"]   = losses[-1]
        else:
            era["mean_loss"] = era["min_loss"] = era["start_loss"] = era["end_loss"] = None
        if era["evals"]:
            tas = [ev["data"]["test_acc"]  for ev in era["evals"]]
            tls = [ev["data"]["test_loss"] for ev in era["evals"]]
            era["best_test_acc"]  = max(tas)
            era["best_test_loss"] = min(tls)
            era["final_test_acc"] = tas[-1]
        else:
            era["best_test_acc"] = era["best_test_loss"] = era["final_test_acc"] = None
    return eras


def build_report_markdown():
    log = state["event_log"]
    if log is None: return "_No experiment yet. Start a run from the Live tab._"
    eras = analyze_experiment(log)
    if not eras: return "_No data yet._"

    lines = ["# Training Experiment Report", ""]

    total_steps = max(e["end_step"] for e in eras)
    total_time  = max(e["end_t"]    for e in eras)
    final_evals = log.filter("eval")
    final_acc = final_evals[-1]["data"]["test_acc"] if final_evals else None
    best_acc  = max((ev["data"]["test_acc"] for ev in final_evals), default=None)

    lines += ["## Overview", "",
              f"- **Total steps**: {total_steps:,}",
              f"- **Wall time**: {total_time:.1f} s",
              f"- **Configuration phases**: {len(eras)}"]
    if final_acc is not None: lines.append(f"- **Final test accuracy**: `{final_acc:.4f}`")
    if best_acc  is not None: lines.append(f"- **Best test accuracy**: `{best_acc:.4f}`")
    lines.append("")

    if state.get("spec"):
        lines += ["## Architecture", "", "```json",
                  json.dumps(state["spec"], indent=2), "```", ""]

    lines += ["## Timeline", "",
              "| Phase | Config | Step range | Mean train loss | Best test acc |",
              "|---|---|---|---|---|"]
    for i, era in enumerate(eras):
        ml  = f"{era['mean_loss']:.4f}"     if era['mean_loss']     is not None else "-"
        bta = f"{era['best_test_acc']:.4f}" if era['best_test_acc'] is not None else "-"
        lines.append(f"| {i+1} | `{format_config(era['config'])}` | "
                     f"{era['start_step']} - {era['end_step']} | {ml} | {bta} |")
    lines.append("")

    transitions = [(i, e) for i, e in enumerate(eras) if e["transition_in"]]
    if transitions:
        lines += ["## Transitions and their impact", ""]
        for i, era in transitions:
            prev = eras[i-1]; t = era["transition_in"]
            lines.append(f"### #{i}: at step {t['step']} (t={t['t']:.1f}s)")
            lines.append("")
            if t["type"] == "optimizer_swap":
                lines.append(f"- **Optimizer swap**: `{format_config(t['data']['from'])}` "
                             f"-> `{format_config(t['data']['to'])}`")
            else:
                lines.append(f"- **LR change**: `{t['data']['from']:.4g}` -> `{t['data']['to']:.4g}`")

            if prev["mean_loss"] is not None and era["mean_loss"] is not None:
                d = era["mean_loss"] - prev["mean_loss"]
                arrow = "improved" if d < 0 else "worsened"
                lines.append(f"- Mean train loss: `{prev['mean_loss']:.4f}` -> "
                             f"`{era['mean_loss']:.4f}` ({arrow} by {abs(d):.4f})")
            if prev["best_test_acc"] is not None and era["best_test_acc"] is not None:
                d = era["best_test_acc"] - prev["best_test_acc"]
                arrow = "improved" if d > 0 else "worsened"
                lines.append(f"- Best test acc: `{prev['best_test_acc']:.4f}` -> "
                             f"`{era['best_test_acc']:.4f}` ({arrow} by {abs(d):.4f})")
            lines.append("")
    return "\n".join(lines)


def export_html_report(path=None):
    if path is None:
        path = os.path.join(ensure_output_dir(), "training_report.html")
    if state["event_log"] is None: return None
    import markdown as md_lib
    body = md_lib.markdown(build_report_markdown(), extensions=["tables", "fenced_code"])
    chart_html = build_figure().to_html(include_plotlyjs="cdn", full_html=False)
    css = (
        "body{font-family:-apple-system,system-ui,sans-serif;max-width:920px;"
        "margin:2em auto;padding:0 1em;line-height:1.55;color:#222}"
        "code{background:#f5f5f5;padding:2px 5px;border-radius:3px;font-size:.92em}"
        "pre{background:#f5f5f5;padding:1em;border-radius:4px;overflow-x:auto}"
        "pre code{background:none;padding:0}"
        "table{border-collapse:collapse;margin:1em 0;width:100%}"
        "th,td{border:1px solid #ddd;padding:.55em .9em;text-align:left}th{background:#f5f5f5}"
        "h1{border-bottom:2px solid #eee;padding-bottom:.4em}"
        "h2{border-bottom:1px solid #eee;padding-bottom:.3em;margin-top:1.8em}"
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Training Report</title><style>" + css + "</style></head><body>"
        + body + "<h2>Training Curves</h2>" + chart_html + "</body></html>"
    )
    with open(path, "w") as f:
        f.write(html)
    return path


def export_event_log(path=None):
    if path is None:
        path = os.path.join(ensure_output_dir(), "event_log.json")
    if state["event_log"] is None: return None
    state["event_log"].save(path)
    return path
