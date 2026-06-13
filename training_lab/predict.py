"""Single-image inference against the currently trained model.

Reuses the same normalization as the test set. Uploaded images are resized to
32x32 to match CIFAR-10 — note that real-world photos are far from CIFAR's tiny,
low-res domain, so predictions on arbitrary images are often unreliable.
"""
import torchvision.transforms as T
import torch

from .config import CIFAR_MEAN, CIFAR_STD, CLASSES
from .state import is_running, state

# Resize to CIFAR size, then the same ToTensor + Normalize as test_tf.
_infer_tf = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def predict_image(pil_image):
    """Return ({class_name: probability, ...}, note_str) for a PIL image.

    Returns (None, message) when prediction can't run.
    """
    model = state["model"]
    if model is None:
        return None, "No trained model yet — run training on the Live tab first."
    if pil_image is None:
        return None, "Upload an image to classify."

    # Avoid racing the training thread's forward pass on BN/Dropout mode.
    controls = state["controls"]
    if is_running() and not (controls and controls.paused):
        return None, "Pause or stop training before predicting (the model is mid-update)."

    device = state["device"]
    x = _infer_tf(pil_image.convert("RGB")).unsqueeze(0).to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu()
    finally:
        model.train(was_training)

    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    top = max(confidences, key=confidences.get)
    note = f"Top prediction: **{top}** ({confidences[top] * 100:.1f}% confidence)"
    return confidences, note
