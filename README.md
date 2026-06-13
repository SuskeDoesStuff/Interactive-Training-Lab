# Interactive Training Lab

A live, visual PyTorch training dashboard for CIFAR-10. Build a model from a
layer spec, watch it train in real time, hot-swap optimizers and learning rates
mid-run, then replay the whole experiment and generate a report — all from a
Gradio web UI.

```bash
python main.py
```

That prints a local URL and a public `gradio.live` link, and opens the dashboard.

## Features

- **Configurable model.** The architecture is a "sandwich": fixed input + fixed
  classifier head, with a fully customizable middle defined as a list of layer
  dicts. Every config is validated at build time via a dummy forward pass, so it
  either compiles or raises a clear, specific error.
- **Live training.** Loss and accuracy curves stream in as the model trains,
  with test-set evaluations marked at each epoch.
- **Hot-swapping.** Change the learning rate, swap the optimizer (Adam ↔ SGD ↔
  RMSprop ↔ …), or extend the epoch count mid-run. Changes take effect on the
  next training step, and every change is annotated on the chart.
- **Optional LR scheduling.** Enable `ReduceLROnPlateau` to anneal the learning
  rate on the test-loss plateau; reductions are logged and annotated just like
  manual changes.
- **Replay.** Scrub a time slider to replay the experiment from the start,
  watching the curves grow and every optimizer/LR change appear in order.
- **Predict.** After training, upload an image and see the model's class
  probabilities. Images are resized to 32x32 to match CIFAR-10.
- **Reports.** Generate a written summary that slices the run into configuration
  "phases" and quantifies how each change moved train loss and test accuracy.
  Exports to self-contained HTML (with the interactive chart embedded) and raw
  JSON (the full event log).

## Quick start

```bash
git clone <your-repo-url>
cd interactive-training-lab
pip install -r requirements.txt
python main.py
```

A CUDA GPU is strongly recommended. On the first run, CIF-10 downloads (~170 MB)
into `./data`. Reports and event logs are written to `./outputs`.

### Command-line options

```
python main.py                 # local + public gradio.live link
python main.py --no-share      # local only (no public link)
python main.py --port 8080     # custom port
python main.py --host 0.0.0.0  # bind on all interfaces
python main.py --verbose       # show all warnings (don't silence noise)
```

If the preferred port is busy, the app automatically falls back to the next
free port instead of failing. Press Ctrl-C in the terminal to stop the server
cleanly and release the port.

For UI development with auto-reload on file changes, use the Gradio CLI:

​```bash
gradio dev.py
​```

### Running on Google Colab

Set the runtime to a GPU (Runtime → Change runtime type → T4 GPU), then:

```python
!git clone <your-repo-url>
%cd interactive-training-lab
!python3 -m venv .venv
!source .venv/bin/activate
!pip install -q -r requirements.txt
!python main.py
```

The public `gradio.live` URL in the output opens the dashboard in a new tab.

## How to use the dashboard

1. **Live tab.** Optionally pick a preset (it loads into the architecture
   editor, which you can hand-edit). Set the optimizer, learning rate, and
   epochs; optionally enable the scheduler. Hit **Start**.
2. While it trains, use **Live adjust** to change the LR, swap the optimizer, or
   raise the epoch count. **Pause** / **Resume** / **Stop** as needed.
3. **Replay tab.** Click **Load**, then drag the time slider to replay the run.
4. **Report tab.** Click **Generate report** to produce the summary and the
   HTML/JSON downloads.
5. **Predict tab.** Pause or stop training, upload an image, and hit
   **Predict** to see the model's class probabilities.

## The layer spec format

The model middle is a list of dicts. Available layer types:

| `type` | Parameters |
|---|---|
| `conv2d` | `out_channels`, `kernel_size`, `stride`, `padding` |
| `batchnorm2d` / `batchnorm1d` | — (channels inferred) |
| `maxpool2d` / `avgpool2d` | `kernel_size`, `stride` |
| `adaptive_avgpool2d` | `output_size` |
| `flatten` | — |
| `linear` | `out_features` |
| `dropout` / `dropout2d` | `p` |
| `activation` | `fn`: `relu`, `gelu`, `leaky_relu`, `silu`, `tanh`, `sigmoid`, `elu` |

The input (`3×32×32`), the final classifier `Linear(…→10)`, and the
`CrossEntropyLoss` are fixed. If the middle ends in a 4D tensor, the head inserts
`AdaptiveAvgPool2d` before flattening; if it already ends 2D (e.g. after an
explicit `flatten` + `linear` stack), the head just adds the final layer.

Example:

```python
[
    {"type": "conv2d", "out_channels": 32, "kernel_size": 3},
    {"type": "batchnorm2d"},
    {"type": "activation", "fn": "relu"},
    {"type": "maxpool2d", "kernel_size": 2},
]
```

## Project structure

```
interactive-training-lab/
├── main.py                 # CLI entrypoint: python main.py
├── requirements.txt
├── training_lab/
│   ├── config.py           # device, seed, CIFAR stats, paths
│   ├── data.py             # transforms + loaders
│   ├── layers.py           # layer builders + shape validation
│   ├── model.py            # SandwichModel
│   ├── specs.py            # default_spec, article_spec, PRESETS
│   ├── controls.py         # shared, mutable training config
│   ├── events.py           # append-only event log (source of truth)
│   ├── trainer.py          # pausable trainer + hot-swap + scheduler
│   ├── charts.py           # Plotly figure builders
│   ├── reporting.py        # phase analysis + report export
│   ├── predict.py          # single-image inference
│   ├── utils.py            # small shared helpers
│   ├── state.py            # app-state singleton
│   └── app.py              # Gradio UI + handlers + main()
```

The design keeps a clean separation: the UI writes to `Controls`, the trainer
reads it each step, and the trainer writes to the `EventLog`, which the chart,
replay, and report all read from. Nothing in the data path calls the UI directly.

## Credits

The **"Article VGG-style"** preset (`article_spec` in
[`training_lab/specs.py`](training_lab/specs.py)) is adapted from Ander's Medium
post, *["The best CNN for CIFAR10 from scratch (93% accuracy)"](https://aaqumon.medium.com/the-best-cnn-for-cifar10-from-scratch-93-accuracy-bde35e17fca6)*
(2024). The original defines an `AdvancedCNN` (a VGG-style network with nine
convolutional layers and a three-layer fully-connected head); it has been
translated here into the layer-spec format. Full credit to the original author
for the architecture. Reproducing its reported accuracy also depends on the
article's training recipe (its specific augmentation, SGD + LR annealing, and
~100 epochs), which differs from the defaults here.

## License

MIT — see [LICENSE](LICENSE).
