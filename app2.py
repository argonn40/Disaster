"""Streamlit interface: upload a satellite/aerial image and classify the disaster.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import load_model_for_inference, predict_pil_image  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "resnet18" / "best.pt"

# Keyed by lower-cased class name, so both "Fire" (AIDERv2) and "Wildfire"
# folder names work.
CLASS_INFO = {
    "earthquake": ("🏚️", "Collapsed buildings, rubble and structural damage after an earthquake."),
    "flood": ("🌊", "Standing water covering streets, fields, buildings or vehicles."),
    "fire": ("🔥", "Active flames, burning vegetation or large smoke plumes."),
    "wildfire": ("🔥", "Active flames, burning vegetation or large smoke plumes."),
    "normal": ("🛰️", "An aerial scene with no visible disaster."),
}

st.set_page_config(
    page_title="Satellite Disaster Classifier",
    page_icon="🛰️",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading the model ...")
def get_model(checkpoint_path: str, device_name: str):
    """Load (and cache) the trained network."""
    return load_model_for_inference(checkpoint_path, device=device_name)


def describe(class_name: str) -> str:
    emoji, text = CLASS_INFO.get(class_name.lower(), ("🛰️", ""))
    return f"{emoji} **{class_name}** — {text}"


def main() -> None:
    st.title("🛰️ Satellite Disaster Image Classifier")
    st.caption(
        "ResNet-18 trained from scratch on the AIDERv2 aerial imagery dataset: "
        "Earthquake · Flood · Fire · Normal"
    )

    with st.sidebar:
        st.header("Model")
        checkpoint_input = st.text_input(
            "Checkpoint file", value=str(DEFAULT_CHECKPOINT)
        )
        default_index = 0 if torch.cuda.is_available() else 1
        device_name = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=default_index,
            help="Use cuda (your RTX 3050) for fastest inference.",
        )
        use_tta = st.checkbox(
            "Test-time augmentation",
            value=True,
            help="Average the prediction with a horizontally flipped copy. "
            "Slightly more accurate, slightly slower.",
        )
        st.divider()
        st.caption(f"PyTorch {torch.__version__}")

    checkpoint_path = Path(checkpoint_input).expanduser()
    if not checkpoint_path.exists():
        st.error(
            f"No checkpoint found at `{checkpoint_path}`.\n\n"
            "Train the model first (see README.md):"
        )
        st.code(
            "python download_data.py\n"
            "python train.py --data-dir data/aider --arch resnet18 --epochs 30",
            language="bash",
        )
        st.stop()

    try:
        model, class_names, img_size, metadata = get_model(
            str(checkpoint_path), device_name
        )
    except Exception as error:  # noqa: BLE001 - show the real message to the user
        st.error(f"Could not load the checkpoint:\n\n`{error}`")
        st.stop()

    with st.sidebar:
        parameters = sum(p.numel() for p in model.parameters())
        st.markdown(f"**Architecture:** {metadata.get('arch')}")
        st.markdown(f"**Input size:** {img_size} × {img_size}")
        st.markdown(f"**Parameters:** {parameters / 1e6:.2f} M")
        if metadata.get("epoch"):
            st.markdown(f"**Best epoch:** {metadata['epoch']}")
        metrics = metadata.get("metrics") or {}
        if metrics:
            accuracy = metrics.get("accuracy")
            macro_f1 = metrics.get("macro_f1")
            if accuracy is not None:
                st.metric("Validation accuracy", f"{accuracy * 100:.2f}%")
            if macro_f1 is not None:
                st.metric("Macro F1", f"{macro_f1 * 100:.2f}%")

    with st.expander("What do the four classes mean?"):
        for class_name in class_names:
            st.markdown(describe(class_name))

    uploaded = st.file_uploader(
        "Upload a satellite / aerial image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
    )

    if uploaded is None:
        st.info("Upload an image to classify it. Dataset images work best.")
        return

    try:
        image = Image.open(uploaded)
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be read as an image.")
        return

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption=uploaded.name)

    with right:
        result = predict_pil_image(
            model,
            image,
            class_names,
            img_size=img_size,
            device=device_name,
            tta=use_tta,
        )
        label = result["label"]
        confidence = result["confidence"]

        st.caption("Prediction")
        st.markdown(f"### {describe(label).split('—')[0].strip()}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")

        if confidence < 0.55:
            st.warning(
                "Low confidence — the image may be unlike the training data "
                "(for example a map, a webcam photo of a street, or heavy cloud)."
            )

        probabilities = pd.DataFrame(
            {"Probability": [result["probabilities"][name] for name in class_names]},
            index=class_names,
        )
        st.bar_chart(probabilities)

    st.subheader("Class probabilities")
    table = pd.DataFrame(
        {
            "Class": class_names,
            "Probability": [
                f"{result['probabilities'][name] * 100:.2f}%" for name in class_names
            ],
        }
    )
    st.dataframe(table, hide_index=True)

    st.caption(
        "Educational model: trained on aerial disaster imagery. "
        "Do not use it for real emergency-response decisions."
    )


if __name__ == "__main__":
    main()
