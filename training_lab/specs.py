"""Architecture presets in the sandwich layer-spec format.

default_spec: a small, fast CNN (~67k params) for quick iteration.

article_spec: a VGG-style net (~108M params) adapted from Ander's Medium post,
'The best CNN for CIFAR10 from scratch (93% accuracy)' (2024). The original
defines an AdvancedCNN with 9 conv layers and a 3-layer FC head; it is
translated here into the layer-spec format. Full credit to the author.
Source: https://aaqumon.medium.com/the-best-cnn-for-cifar10-from-scratch-93-accuracy-bde35e17fca6
"""

default_spec = [
    {"type": "conv2d",      "out_channels": 32, "kernel_size": 3},
    {"type": "batchnorm2d"},
    {"type": "activation",  "fn": "relu"},
    {"type": "conv2d",      "out_channels": 32, "kernel_size": 3},
    {"type": "batchnorm2d"},
    {"type": "activation",  "fn": "relu"},
    {"type": "maxpool2d",   "kernel_size": 2},

    {"type": "conv2d",      "out_channels": 64, "kernel_size": 3},
    {"type": "batchnorm2d"},
    {"type": "activation",  "fn": "relu"},
    {"type": "conv2d",      "out_channels": 64, "kernel_size": 3},
    {"type": "batchnorm2d"},
    {"type": "activation",  "fn": "relu"},
    {"type": "maxpool2d",   "kernel_size": 2},

    {"type": "dropout2d",   "p": 0.25},
]

# VGG-style net from the "93% CIFAR10 from scratch" article, translated to the
# sandwich spec. ~108M params (the FC head is ~93% of them). The FC layers MUST
# be explicit here: if the middle ended in a 4D tensor, SandwichModel's head
# would insert AdaptiveAvgPool2d and silently discard this design. Ending the
# middle in a 2D tensor (after flatten + FC) means the head just adds the final
# Linear(4096 -> 10).
article_spec = [
    # Block 1
    {"type": "conv2d", "out_channels": 64,  "kernel_size": 3},
    {"type": "batchnorm2d"},
    {"type": "activation", "fn": "relu"},
    {"type": "conv2d", "out_channels": 128, "kernel_size": 3},
    {"type": "activation", "fn": "relu"},
    {"type": "conv2d", "out_channels": 128, "kernel_size": 3},
    {"type": "activation", "fn": "relu"},
    {"type": "maxpool2d", "kernel_size": 2},

    # Block 2
    {"type": "conv2d", "out_channels": 256, "kernel_size": 3},
    {"type": "batchnorm2d"},
    {"type": "activation", "fn": "relu"},
    {"type": "conv2d", "out_channels": 256, "kernel_size": 3},
    {"type": "activation", "fn": "relu"},
    {"type": "conv2d", "out_channels": 256, "kernel_size": 3},
    {"type": "activation", "fn": "relu"},
    {"type": "maxpool2d", "kernel_size": 2},
    {"type": "dropout2d", "p": 0.2},

    # Block 3
    {"type": "conv2d", "out_channels": 512, "kernel_size": 3},
    {"type": "batchnorm2d"},
    {"type": "activation", "fn": "relu"},
    {"type": "conv2d", "out_channels": 512, "kernel_size": 3},
    {"type": "activation", "fn": "relu"},
    {"type": "conv2d", "out_channels": 512, "kernel_size": 3},
    {"type": "activation", "fn": "relu"},
    {"type": "maxpool2d", "kernel_size": 2},
    {"type": "dropout2d", "p": 0.2},

    # FC head (explicit; head appends the final Linear -> 10)
    {"type": "flatten"},
    {"type": "linear", "out_features": 8192},
    {"type": "activation", "fn": "relu"},
    {"type": "linear", "out_features": 4096},
    {"type": "activation", "fn": "relu"},
    {"type": "dropout", "p": 0.2},
]

ARTICLE_URL = "https://aaqumon.medium.com/the-best-cnn-for-cifar10-from-scratch-93-accuracy-bde35e17fca6"
ARTICLE_CREDIT = (
    "The 'Article VGG-style' preset is adapted from Ander's Medium post "
    "'The best CNN for CIFAR10 from scratch (93% accuracy)'."
)

PRESETS = {
    "Default (small, ~67k params)": default_spec,
    "Article VGG-style (~108M params)": article_spec,
}
