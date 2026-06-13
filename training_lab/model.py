"""SandwichModel: fixed input + customizable middle + adaptive classifier head."""
import torch
import torch.nn as nn

from .layers import LAYER_BUILDERS

def _compute_out_shape(module, in_shape):
    """Run a dummy forward to get output shape. Eval mode avoids BN batch-size-1 issues."""
    was_training = module.training
    module.eval()
    with torch.no_grad():
        out = module(torch.zeros(in_shape))
    module.train(was_training)
    return tuple(out.shape)


class SandwichModel(nn.Module):
    def __init__(self, middle_spec, in_shape=(3, 32, 32), num_classes=10):
        super().__init__()
        self.middle_spec = list(middle_spec)
        self.in_shape = tuple(in_shape)
        self.num_classes = num_classes

        shape = (1,) + self.in_shape
        layers = []
        self._layer_shapes = [shape]

        for i, spec in enumerate(self.middle_spec):
            layer_type = spec.get("type")
            if layer_type not in LAYER_BUILDERS:
                raise ValueError(
                    f"Layer {i}: unknown type '{layer_type}'. "
                    f"Available: {list(LAYER_BUILDERS)}"
                )
            try:
                module = LAYER_BUILDERS[layer_type](spec, shape)
                new_shape = _compute_out_shape(module, shape)
            except Exception as e:
                raise ValueError(f"Layer {i} ({layer_type}): {e}") from e

            if len(new_shape) == 4 and (new_shape[2] < 1 or new_shape[3] < 1):
                raise ValueError(
                    f"Layer {i} ({layer_type}): output spatial dims {new_shape[2:]} "
                    f"are invalid - too much downsampling for input {self.in_shape}."
                )

            layers.append(module)
            shape = new_shape
            self._layer_shapes.append(shape)

        self.middle = nn.Sequential(*layers)

        head_layers = []
        if len(shape) == 4:
            head_layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
            features = shape[1]
        elif len(shape) == 2:
            features = shape[1]
        else:
            raise ValueError(f"Middle produced unsupported shape {shape}; must be 2D or 4D.")
        head_layers.append(nn.Linear(features, num_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.head(self.middle(x))

    def summary(self):
        print(f"Input: (B, {', '.join(str(d) for d in self.in_shape)})")
        print("-" * 72)
        print(f"{'#':<3} {'Layer':<24} {'Output Shape':<28} {'Params':>12}")
        print("-" * 72)
        total = 0
        for i, (spec, module, shape) in enumerate(
            zip(self.middle_spec, self.middle, self._layer_shapes[1:])
        ):
            n = sum(p.numel() for p in module.parameters())
            total += n
            label = spec["type"]
            for k in ("out_channels", "out_features", "fn", "kernel_size", "p"):
                if k in spec:
                    label += f"({spec[k]})"
                    break
            shape_str = "(" + ", ".join(str(d) for d in shape) + ")"
            print(f"{i:<3} {label:<24} {shape_str:<28} {n:>12,}")
        head_params = sum(p.numel() for p in self.head.parameters())
        total += head_params
        print("-" * 72)
        print(f"{'':<3} {'HEAD (pool+linear)':<24} {f'(B, {self.num_classes})':<28} {head_params:>12,}")
        print("-" * 72)
        print(f"Total trainable params: {total:,}")
