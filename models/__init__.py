import os

from models.vanilla_gan import (
    load_generator as _load_vanilla,
    generate as _gen_vanilla,
)
from models.dcgan import (
    load_generator as _load_dcgan,
    generate as _gen_dcgan,
)
from models.cgan import (
    load_generator as _load_cgan,
    generate as _gen_cgan,
)

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
# To add a new model (e.g. CGAN, DCGAN):
#   1. Create  models/<model_name>.py  with  load_generator()  and  generate()
#   2. Import the functions above
#   3. Add an entry to MODEL_REGISTRY below
#   4. Drop the checkpoint into  saved_models/
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "Vanilla GAN": {
        "checkpoint": "vanilla_gan.pth",
        "load_fn": _load_vanilla,
        "generate_fn": _gen_vanilla,
        "description": "Fully-connected GAN trained on MNIST (28×28 grayscale)",
    },
    "DCGAN": {
        "checkpoint": "dcgan.pth",
        "load_fn": _load_dcgan,
        "generate_fn": _gen_dcgan,
        "description": "Deep Convolutional GAN trained on MNIST (28×28 grayscale)",
        "conditional": False,
    },
    "CGAN": {
        "checkpoint": "cgan.pth",
        "load_fn": _load_cgan,
        "generate_fn": _gen_cgan,
        "description": "Conditional GAN trained on MNIST — generates a specific digit (0–9)",
        "conditional": True,
    },
}

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")


def get_available_models():
    """Return only registry entries whose checkpoint files actually exist."""
    available = {}
    for name, cfg in MODEL_REGISTRY.items():
        path = os.path.join(SAVED_MODELS_DIR, cfg["checkpoint"])
        if os.path.isfile(path):
            available[name] = {**cfg, "checkpoint_path": path}
    return available
