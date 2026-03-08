import streamlit as st
import torch
import numpy as np
from PIL import Image

from models import get_available_models

# ---- Page config ----
st.set_page_config(page_title="GAN Image Generator", layout="wide")
st.title("GAN Image Generator")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Discover available models ----
available = get_available_models()

if not available:
    st.error("No trained models found in saved_models/. Train a model first.")
    st.stop()

# ---- Sidebar: model selection ----
model_name = st.sidebar.selectbox("Select Model", list(available.keys()))
model_cfg = available[model_name]
st.sidebar.caption(model_cfg["description"])
st.sidebar.markdown(f"**Device:** {device}")


# ---- Load generator (cached across reruns) ----
@st.cache_resource
def load_model(name, path, _load_fn):
    return _load_fn(path, device)


generator = load_model(model_name, model_cfg["checkpoint_path"], model_cfg["load_fn"])


# ---- Helper: tensor → displayable numpy ----
def to_display(img_tensor):
    """Convert a single CHW tensor in [0,1] to a numpy array for st.image."""
    img = img_tensor.cpu().numpy()
    if img.shape[0] == 1:
        return img.squeeze(0)          # grayscale → HW
    return np.transpose(img, (1, 2, 0))  # RGB → HWC


# ---- Tabs ----
tab_upload, tab_quick = st.tabs(["Upload & Generate", "Quick Generate"])

# -- Tab 1: Upload images → generate same count --
with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload image(s) — one generated image per upload",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        num = len(uploaded_files)
        gen_imgs = model_cfg["generate_fn"](generator, num, device)

        st.subheader(f"Results ({num} image{'s' if num > 1 else ''})")

        for idx, (uf, gi) in enumerate(zip(uploaded_files, gen_imgs)):
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Uploaded")
                st.image(Image.open(uf), use_container_width=True)
            with col2:
                st.caption("Generated")
                st.image(to_display(gi), use_container_width=True, clamp=True)
    else:
        st.info("Upload one or more images to generate GAN outputs.")

# -- Tab 2: Generate N random samples (no upload needed) --
with tab_quick:
    num_gen = st.slider("Number of images to generate", 1, 16, 4)

    if st.button("Generate"):
        gen_imgs = model_cfg["generate_fn"](generator, num_gen, device)

        cols = st.columns(min(num_gen, 4))
        for idx, gi in enumerate(gen_imgs):
            with cols[idx % 4]:
                st.image(to_display(gi), use_container_width=True, clamp=True)
