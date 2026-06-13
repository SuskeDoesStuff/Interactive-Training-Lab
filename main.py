"""Entrypoint: `python main.py` launches the dashboard and prints the Gradio link.

Examples:
    python main.py                 # local + public gradio.live link
    python main.py --no-share      # local only (no public link)
    python main.py --port 8080     # custom port (auto-falls back if busy)
    python main.py --verbose       # show all warnings (don't silence noise)

For UI development with auto-reload, use the Gradio CLI instead:
    gradio dev.py
"""
import argparse

from training_lab.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive Training Lab — live, visual PyTorch training on CIFAR-10."
    )
    parser.add_argument("--no-share", action="store_true",
                        help="Disable the public gradio.live link (local only).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860,
                        help="Preferred port (default: 7860). Falls back to the "
                             "next free port if busy.")
    parser.add_argument("--debug", action="store_true",
                        help="Launch Gradio in debug mode.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all warnings instead of silencing known noise.")
    args = parser.parse_args()

    main(
        share=not args.no_share,
        server_name=args.host,
        server_port=args.port,
        debug=args.debug,
        verbose=args.verbose,
    )