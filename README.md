# GAN Image Generator

A collection of GAN (Generative Adversarial Network) models with a Streamlit web UI for generating images.

## Project Structure

```
GAN/
├── app.py                  # Streamlit web UI
├── Vanilla_GAN.py          # Training script for Vanilla GAN
├── models/
│   ├── __init__.py         # Model registry & auto-discovery
│   └── vanilla_gan.py      # Generator architecture & loading
├── saved_models/
│   └── vanilla_gan.pth     # Trained Vanilla GAN checkpoint
└── README.md
```

## Models

| Model       | Architecture           | Dataset | Status  |
|-------------|------------------------|---------|---------|
| Vanilla GAN | Fully-connected layers | MNIST   | Ready   |
| CGAN        | Conditional GAN        | MNIST   | -----   |
| DCGAN       | Deep Convolutional GAN | MNIST   | Ready   |

## Setup

### Requirements

- Python 3.10+
- PyTorch
- Streamlit

### Install Dependencies

```bash
pip install torch torchvision streamlit
```

## Usage

### Train a Model

```bash
python Vanilla_GAN.py
```

This trains the Vanilla GAN on MNIST for 500 epochs and saves the checkpoint to `saved_models/vanilla_gan.pth`.

### Launch the Web UI

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Web UI Features

- **Model Selection** — Dropdown in the sidebar lists all available models (only those with a checkpoint in `saved_models/`)
- **Upload & Generate** — Upload one or more images and get a generated image for each upload, displayed side-by-side
- **Quick Generate** — Generate 1–16 random samples without uploading anything

## Adding a New Model

1. Create `models/<model_name>.py` with two functions:
   - `load_generator(checkpoint_path, device)` — loads weights and returns the generator in eval mode
   - `generate(generator, num_images, device, **kwargs)` — returns a tensor of generated images in `[0, 1]` range
2. Import them in `models/__init__.py` and add an entry to `MODEL_REGISTRY`
3. Place the trained checkpoint in `saved_models/`
