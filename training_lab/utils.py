"""Small pure helpers shared across modules (no state, no heavy imports)."""


def ema(values, alpha=0.05):
    """Exponential moving average. Smooths noisy per-step training curves."""
    if not values:
        return []
    s, out = values[0], []
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def format_config(cfg):
    """Render an optimizer config dict as a short human-readable string."""
    if not cfg:
        return "?"
    s = f"{cfg.get('name', '?')} @ lr={cfg.get('lr', 0):.4g}"
    kwargs = cfg.get("kwargs") or {}
    if kwargs:
        s += " (" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"
    return s
