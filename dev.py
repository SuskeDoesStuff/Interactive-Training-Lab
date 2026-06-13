"""UI-development entrypoint for the Gradio CLI's auto-reload mode:

    gradio dev.py

Loads the dataset once at import and exposes a module-level `demo` that the
Gradio CLI watches and hot-reloads. For normal use, prefer `python main.py`.
"""
from training_lab.app import build_demo, quiet_warnings, setup_state

quiet_warnings()
setup_state()
demo = build_demo()