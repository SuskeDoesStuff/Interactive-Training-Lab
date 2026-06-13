"""Gradio dashboard: state-aware status/handlers, the three-tab UI, and the
main() entrypoint that wires data + device and launches the app."""
import json
import threading

import gradio as gr
import torch

from .charts import _build_chart, build_figure, build_figure_at, get_replay_max_t
from .config import get_device, set_seed
from .controls import Controls
from .data import build_loaders
from .events import EventLog
from .model import SandwichModel
from .predict import predict_image
from .reporting import build_report_markdown, export_event_log, export_html_report
from .specs import ARTICLE_CREDIT, ARTICLE_URL, PRESETS, default_spec
from .state import is_running, state
from .trainer import OPTIMIZERS, Trainer


def build_status():
    if state["controls"] is None:
        return "**Not started.**"
    c, log = state["controls"], state["event_log"]
    verb = "Paused" if c.paused and is_running() else ("Running" if is_running() else "Stopped")
    head = (f"**{verb}**  |  epoch {c.epoch}/{c.max_epochs}  |  step {c.step}  "
            f"|  opt: `{c.optimizer_name}`  |  lr: `{c.lr:.4g}`")
    tail = ""
    ts = log.filter("train_step")
    if ts:
        d = ts[-1]["data"]
        tail += f"train loss `{d['loss']:.4f}`, acc `{d['acc']:.3f}`"
    ev = log.filter("eval")
    if ev:
        d = ev[-1]["data"]
        tail += f"  -  test loss `{d['test_loss']:.4f}`, acc `{d['test_acc']:.3f}`"
    return head + ("\n\n" + tail if tail else "")


def build_events_text():
    log = state["event_log"]
    if log is None: return ""
    interesting = log.filter("optimizer_swap", "lr_change", "paused", "resumed",
                             "epoch_end", "eval", "training_start", "training_end")
    lines = []
    for e in interesting[-15:]:
        tag = f"[t={e['t']:6.1f}s step={(e['step'] or 0):5d}]"
        d = e["data"]
        if e["type"] == "optimizer_swap":
            msg = f"SWAP {d['from']['name']} -> {d['to']['name']}  lr={d['to']['lr']:.4g}"
        elif e["type"] == "lr_change":
            msg = f"LR  {d['from']:.4g} -> {d['to']:.4g}"
        elif e["type"] == "eval":
            msg = f"EVAL test_loss={d['test_loss']:.4f}  test_acc={d['test_acc']:.4f}"
        else:
            msg = e["type"].upper()
        lines.append(f"{tag} {msg}")
    return "\n".join(lines) if lines else "(no events yet)"


def refresh():
    return build_figure(), build_status(), build_events_text()


def start_training(spec_json, optimizer, lr, max_epochs, momentum, weight_decay,
                   use_scheduler, sch_factor, sch_patience, sch_min_lr):
    if is_running():
        return "Already running - Stop first."
    try:
        spec = json.loads(spec_json)
        assert isinstance(spec, list)
    except Exception as e:
        return f"Spec parse error: {e}"
    try:
        model = SandwichModel(spec).to(device)
    except ValueError as e:
        return f"Model build error: {e}"

    kwargs = {}
    if optimizer == "sgd" and momentum > 0: kwargs["momentum"] = float(momentum)
    if weight_decay > 0:                    kwargs["weight_decay"] = float(weight_decay)

    scheduler_cfg = {"factor": float(sch_factor),
                     "patience": int(sch_patience),
                     "min_lr": float(sch_min_lr)}

    state["spec"]      = spec
    state["model"]     = model
    state["controls"]  = Controls(optimizer=optimizer, lr=float(lr),
                                  max_epochs=int(max_epochs), optimizer_kwargs=kwargs,
                                  use_scheduler=bool(use_scheduler),
                                  scheduler_cfg=scheduler_cfg)
    state["event_log"] = EventLog()
    state["trainer"]   = Trainer(model, state["train_loader"], state["test_loader"],
                                 state["controls"], state["event_log"],
                                 device=state["device"], log_every=5)
    state["thread"]    = threading.Thread(target=state["trainer"].run, daemon=True)
    state["thread"].start()
    sch = f" + ReduceLROnPlateau(factor={sch_factor}, patience={sch_patience})" if use_scheduler else ""
    return f"Training started.{sch}"


def do_pause():
    if state["controls"]: state["controls"].pause()
    return build_status()

def do_resume():
    if state["controls"]: state["controls"].resume()
    return build_status()

def do_stop():
    if state["controls"]: state["controls"].stop()
    return build_status()

def apply_lr(new_lr):
    if state["controls"]: state["controls"].set_lr(float(new_lr))
    return build_status()

def swap_optimizer_action(new_opt, new_lr, momentum, weight_decay):
    if not state["controls"]: return "No training running."
    kwargs = {}
    if new_opt == "sgd" and momentum > 0: kwargs["momentum"] = float(momentum)
    if weight_decay > 0:                  kwargs["weight_decay"] = float(weight_decay)
    state["controls"].set_optimizer(new_opt, lr=float(new_lr), **kwargs)
    return build_status()

def apply_epochs(new_n):
    if state["controls"]: state["controls"].set_max_epochs(int(new_n))
    return build_status()


def load_replay():
    max_t = get_replay_max_t()
    if max_t == 0:
        return (gr.update(minimum=0, maximum=1, value=0),
                _build_chart([], "No experiment recorded yet"), "_No data._")
    fig, status = build_figure_at(max_t)
    return (gr.update(minimum=0, maximum=max_t, value=max_t, step=max(0.1, max_t/200)),
            fig, status)


def generate_report():
    md = build_report_markdown()
    return md, export_html_report(), export_event_log()


def load_preset(name):
    return json.dumps(PRESETS[name], indent=2)


def build_demo():
    with gr.Blocks(title="Training Lab") as demo:
        gr.Markdown("# Training Lab")

        with gr.Tabs():

            with gr.Tab("Live"):
                live_status = gr.Markdown("**Not started.**")
                with gr.Row():
                    with gr.Column(scale=1):
                        preset = gr.Dropdown(choices=list(PRESETS),
                                             value="Default (small, ~67k params)",
                                             label="Preset (loads into the editor below)")
                        gr.Markdown(f"<sub>{ARTICLE_CREDIT} "
                                    f"[Source]({ARTICLE_URL}).</sub>")
                        with gr.Accordion("Model architecture (edit before Start)", open=False):
                            spec_input = gr.Code(value=json.dumps(default_spec, indent=2),
                                                 language="json", lines=18)
                        gr.Markdown("### Initial config")
                        optimizer = gr.Dropdown(choices=list(OPTIMIZERS), value="adam", label="Optimizer")
                        with gr.Row():
                            lr = gr.Number(value=1e-3, label="Learning rate", precision=6)
                            max_epochs = gr.Number(value=5, label="Max epochs", precision=0)
                        with gr.Row():
                            momentum = gr.Slider(0, 0.99, value=0.9, step=0.01, label="Momentum (SGD)")
                            weight_decay = gr.Number(value=0.0, label="Weight decay")

                        gr.Markdown("### LR scheduler (set before Start)")
                        use_scheduler = gr.Checkbox(value=False, label="ReduceLROnPlateau (on test loss)")
                        with gr.Row():
                            sch_factor   = gr.Number(value=0.1,  label="factor")
                            sch_patience = gr.Number(value=10,   label="patience", precision=0)
                            sch_min_lr   = gr.Number(value=1e-5, label="min lr", precision=8)

                        with gr.Row():
                            start_btn  = gr.Button("Start",  variant="primary")
                            pause_btn  = gr.Button("Pause")
                            resume_btn = gr.Button("Resume")
                            stop_btn   = gr.Button("Stop", variant="stop")

                        gr.Markdown("### Live adjust")
                        with gr.Row():
                            live_lr = gr.Number(value=1e-4, label="New LR")
                            apply_lr_btn = gr.Button("Apply LR")
                        with gr.Row():
                            live_opt = gr.Dropdown(choices=list(OPTIMIZERS), value="sgd", label="Swap to")
                            swap_btn = gr.Button("Swap optimizer")
                        with gr.Row():
                            new_epochs = gr.Number(value=10, label="New max epochs", precision=0)
                            apply_epochs_btn = gr.Button("Update epochs")

                    with gr.Column(scale=2):
                        live_chart = gr.Plot()
                        events_text = gr.Textbox(label="Event feed", lines=10, interactive=False)

                timer = gr.Timer(1.5)
                timer.tick(refresh, outputs=[live_chart, live_status, events_text])

                preset.change(load_preset, inputs=[preset], outputs=[spec_input])

                start_btn.click(start_training,
                                inputs=[spec_input, optimizer, lr, max_epochs, momentum, weight_decay,
                                        use_scheduler, sch_factor, sch_patience, sch_min_lr],
                                outputs=live_status)
                pause_btn.click(do_pause,   outputs=live_status)
                resume_btn.click(do_resume, outputs=live_status)
                stop_btn.click(do_stop,     outputs=live_status)
                apply_lr_btn.click(apply_lr, inputs=[live_lr], outputs=live_status)
                swap_btn.click(swap_optimizer_action,
                               inputs=[live_opt, live_lr, momentum, weight_decay], outputs=live_status)
                apply_epochs_btn.click(apply_epochs, inputs=[new_epochs], outputs=live_status)

            with gr.Tab("Replay"):
                gr.Markdown("Scrub through the experiment as it happened. Click Load first; "
                            "re-load any time to refresh the time range against the latest log.")
                load_btn = gr.Button("Load current experiment", variant="primary")
                time_slider = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Time (seconds)")
                replay_status = gr.Markdown("_Click Load to start._")
                replay_chart = gr.Plot()

                load_btn.click(load_replay, outputs=[time_slider, replay_chart, replay_status])
                time_slider.change(build_figure_at, inputs=[time_slider],
                                   outputs=[replay_chart, replay_status])

            with gr.Tab("Report"):
                gr.Markdown("Generate a written summary. The HTML embeds the interactive chart; "
                            "the JSON is the raw event log for offline analysis or run comparison.")
                gen_btn = gr.Button("Generate report", variant="primary")
                report_md = gr.Markdown("_No report yet._")
                with gr.Row():
                    html_file = gr.File(label="HTML report")
                    json_file = gr.File(label="Event log (JSON)")
                gen_btn.click(generate_report, outputs=[report_md, html_file, json_file])

            with gr.Tab("Predict"):
                gr.Markdown(
                    "Upload an image to see what the trained model predicts. The image is "
                    "resized to 32x32 to match CIFAR-10, so everyday photos are far outside "
                    "the training domain and predictions on them are often unreliable — "
                    "CIFAR test images or simple cropped objects work best. "
                    "Pause or stop training before predicting."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        predict_img = gr.Image(type="pil", label="Image", height=240)
                        predict_btn = gr.Button("Predict", variant="primary")
                    with gr.Column(scale=1):
                        predict_note = gr.Markdown("_Upload an image and hit Predict._")
                        predict_label = gr.Label(num_top_classes=5, label="Class probabilities")

                predict_btn.click(predict_image, inputs=[predict_img],
                                  outputs=[predict_label, predict_note])
    return demo


def main(share=True, server_name="127.0.0.1", server_port=7860, debug=False):
    set_seed()
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading CIFAR-10 (downloads ~170 MB on first run)...")
    train_loader, test_loader = build_loaders()
    state["train_loader"] = train_loader
    state["test_loader"] = test_loader
    state["device"] = device
    demo = build_demo()
    demo.launch(theme=gr.themes.Soft(), share=share,
                server_name=server_name, server_port=server_port, debug=debug)
