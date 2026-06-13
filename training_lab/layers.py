"""Layer builders for the configurable model. Each builder validates the
incoming tensor shape and returns an nn.Module, raising a clear error when
the shape contract is violated."""
import torch.nn as nn

ACTIVATIONS = {
    "relu":       lambda s: nn.ReLU(inplace=True),
    "gelu":       lambda s: nn.GELU(),
    "leaky_relu": lambda s: nn.LeakyReLU(s.get("negative_slope", 0.01), inplace=True),
    "silu":       lambda s: nn.SiLU(inplace=True),
    "tanh":       lambda s: nn.Tanh(),
    "sigmoid":    lambda s: nn.Sigmoid(),
    "elu":        lambda s: nn.ELU(inplace=True),
}

def _require_dim(shape, n, layer_type, hint=""):
    if len(shape) != n:
        msg = f"{layer_type} expects {n}D input, got shape {tuple(shape)}"
        if hint: msg += f". {hint}"
        raise ValueError(msg)

def _build_conv2d(spec, in_shape):
    _require_dim(in_shape, 4, "conv2d", "Conv2d needs (B,C,H,W); remove a flatten/linear earlier.")
    k = spec.get("kernel_size", 3)
    return nn.Conv2d(
        in_channels=in_shape[1],
        out_channels=spec["out_channels"],
        kernel_size=k,
        stride=spec.get("stride", 1),
        padding=spec.get("padding", k // 2),
    )

def _build_batchnorm2d(spec, in_shape):
    _require_dim(in_shape, 4, "batchnorm2d", "Use batchnorm1d after flatten/linear.")
    return nn.BatchNorm2d(in_shape[1])

def _build_batchnorm1d(spec, in_shape):
    _require_dim(in_shape, 2, "batchnorm1d", "Use batchnorm2d before flatten.")
    return nn.BatchNorm1d(in_shape[1])

def _build_maxpool2d(spec, in_shape):
    _require_dim(in_shape, 4, "maxpool2d")
    k = spec.get("kernel_size", 2)
    return nn.MaxPool2d(kernel_size=k, stride=spec.get("stride", k))

def _build_avgpool2d(spec, in_shape):
    _require_dim(in_shape, 4, "avgpool2d")
    k = spec.get("kernel_size", 2)
    return nn.AvgPool2d(kernel_size=k, stride=spec.get("stride", k))

def _build_adaptive_avgpool2d(spec, in_shape):
    _require_dim(in_shape, 4, "adaptive_avgpool2d")
    return nn.AdaptiveAvgPool2d(spec.get("output_size", 1))

def _build_flatten(spec, in_shape):
    _require_dim(in_shape, 4, "flatten")
    return nn.Flatten()

def _build_linear(spec, in_shape):
    _require_dim(in_shape, 2, "linear", "Add a 'flatten' before linear, or remove conv layers above.")
    return nn.Linear(in_shape[1], spec["out_features"])

def _build_dropout(spec, in_shape):
    return nn.Dropout(p=spec.get("p", 0.5))

def _build_dropout2d(spec, in_shape):
    _require_dim(in_shape, 4, "dropout2d")
    return nn.Dropout2d(p=spec.get("p", 0.5))

def _build_activation(spec, in_shape):
    fn = spec.get("fn", "relu")
    if fn not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{fn}'. Available: {list(ACTIVATIONS)}")
    return ACTIVATIONS[fn](spec)

LAYER_BUILDERS = {
    "conv2d": _build_conv2d,
    "batchnorm2d": _build_batchnorm2d,
    "batchnorm1d": _build_batchnorm1d,
    "maxpool2d": _build_maxpool2d,
    "avgpool2d": _build_avgpool2d,
    "adaptive_avgpool2d": _build_adaptive_avgpool2d,
    "flatten": _build_flatten,
    "linear": _build_linear,
    "dropout": _build_dropout,
    "dropout2d": _build_dropout2d,
    "activation": _build_activation,
}
