"""Satellite disaster classifier - Streamlit UI (v2, drop-in replacement).

Run with:
    streamlit run app.py

Same backend as before (`inference.py`, `runs/resnet18/best.pt`); only the
experience changed: no sidebar, hero header, stat strip, a spotlight result
card with a confidence ring and custom probability bars.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import load_model_for_inference, predict_pil_image  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "resnet18" / "best.pt"

# Colours/emoji per class.  Keyed by lower-case folder name so both "Fire"
# (AIDERv2) and "Wildfire" work.
CLASS_STYLE = {
    "earthquake": {
        "emoji": "🏚️",
        "color": "#f59e0b",
        "blurb": "Collapsed buildings, rubble and structural damage",
    },
    "flood": {
        "emoji": "🌊",
        "color": "#3b82f6",
        "blurb": "Standing water over streets, fields and buildings",
    },
    "fire": {
        "emoji": "🔥",
        "color": "#ef4444",
        "blurb": "Flames, burning vegetation and smoke plumes",
    },
    "wildfire": {
        "emoji": "🔥",
        "color": "#ef4444",
        "blurb": "Flames, burning vegetation and smoke plumes",
    },
    "normal": {
        "emoji": "🛰️",
        "color": "#22c55e",
        "blurb": "An aerial scene with no visible disaster",
    },
}
FALLBACK_STYLE = {"emoji": "🛰️", "color": "#8b5cf6", "blurb": "Aerial imagery"}


def style_for(class_name: str) -> dict:
    return CLASS_STYLE.get(class_name.lower(), FALLBACK_STYLE)


st.set_page_config(
    page_title="Satellite Disaster Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
/* ---------- chrome: no sidebar, no menu clutter ---------- */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none !important;}
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1120px; padding-top: 1.6rem; padding-bottom: 3rem;}

/* ---------- hero ---------- */
.hero {
  position: relative; overflow: hidden;
  border-radius: 24px; padding: 2.3rem 2.3rem 1.7rem; margin-bottom: 1.1rem;
  color: #eef4ff; border: 1px solid rgba(148,163,184,.22);
  background:
    radial-gradient(900px 340px at 6% -30%, rgba(59,130,246,.38), transparent 62%),
    radial-gradient(760px 300px at 96% -10%, rgba(239,68,68,.30), transparent 58%),
    linear-gradient(150deg, #0d1526, #0b1220 68%);
}
.hero::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 38px 38px;
  mask-image: linear-gradient(180deg, rgba(0,0,0,.9), transparent 85%);
}
.hero-badge {
  display: inline-flex; align-items: center; gap: .45rem;
  font-size: .72rem; letter-spacing: .13em; text-transform: uppercase;
  padding: .35rem .7rem; border-radius: 999px;
  color: #bfdbfe; background: rgba(59,130,246,.16);
  border: 1px solid rgba(59,130,246,.4);
}
.hero h1 {
  font-size: 2.45rem; line-height: 1.08; margin: .85rem 0 .55rem;
  font-weight: 700; letter-spacing: -.02em; color: #f8fafc;
}
.hero h1 span {
  background: linear-gradient(92deg, #60a5fa, #f87171 75%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p {margin: 0; max-width: 46rem; font-size: .98rem; line-height: 1.55; color: #c7d2e4;}
.chips {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1.15rem;}
.chip {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .32rem .68rem; border-radius: 999px; font-size: .8rem;
  color: #dbe4f5; background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.14);
}
.dot {width: .48rem; height: .48rem; border-radius: 50%; flex: 0 0 auto;}

/* ---------- stat strip ---------- */
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; margin: .2rem 0 1.15rem;}
.stat {
  border-radius: 16px; padding: .85rem 1rem;
  background: rgba(148,163,184,.10); border: 1px solid rgba(148,163,184,.22);
}
.stat b {display: block; font-size: 1.3rem; line-height: 1.2; font-variant-numeric: tabular-nums;}
.stat span {font-size: .68rem; letter-spacing: .11em; text-transform: uppercase; opacity: .62;}

/* ---------- generic section label ---------- */
.section-title {
  font-size: .7rem; letter-spacing: .16em; text-transform: uppercase;
  opacity: .55; margin: .3rem 0 .6rem;
}

/* ---------- upload / image ---------- */
[data-testid="stFileUploaderDropzone"] {
  border-radius: 18px; border: 1.5px dashed rgba(148,163,184,.45);
  background: rgba(148,163,184,.06); padding: 1.5rem 1.2rem;
}
[data-testid="stFileUploaderDropzone"]:hover {border-color: rgba(96,165,250,.75);}
[data-testid="stFileUploaderDropzone"] button {border-radius: 10px;}
[data-testid="stImage"] img {
  border-radius: 18px; border: 1px solid rgba(148,163,184,.28);
  display: block; width: 100%;
}
.file-meta {font-size: .78rem; opacity: .62; margin-top: .55rem; word-break: break-all;}

/* ---------- spotlight result card ---------- */
.spotlight {
  border-radius: 22px; padding: 1.4rem 1.45rem 1.3rem; color: #e8eefc;
  border: 1px solid rgba(148,163,184,.26);
  background: linear-gradient(160deg, #101a30 0%, #0b1220 62%, #0a101c 100%);
  box-shadow: 0 16px 40px rgba(2,6,23,.38);
}
.verdict {display: flex; align-items: center; gap: 1.15rem;}
.ring {
  width: 124px; height: 124px; flex: 0 0 auto; border-radius: 50%;
  display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(148,163,184,.20) 0);
  box-shadow: 0 0 26px -6px var(--c);
}
.ring-inner {
  width: 98px; height: 98px; border-radius: 50%; background: #0b1220;
  display: grid; place-items: center; line-height: 1.05; text-align: center;
}
.ring-inner b {font-size: 1.32rem; font-variant-numeric: tabular-nums;}
.ring-inner small {font-size: .58rem; letter-spacing: .13em; text-transform: uppercase; opacity: .6;}
.verdict-label {font-size: .68rem; letter-spacing: .15em; text-transform: uppercase; opacity: .58;}
.verdict-class {font-size: 1.9rem; font-weight: 700; letter-spacing: -.01em; margin: .2rem 0 .15rem;}
.verdict-sub {font-size: .84rem; opacity: .72;}

/* ---------- probability bars ---------- */
.bars {margin-top: 1.2rem; display: flex; flex-direction: column; gap: .6rem;}
.bar-row {
  display: grid; grid-template-columns: 112px 1fr 50px;
  align-items: center; gap: .6rem; font-size: .82rem;
}
.bar-name {white-space: nowrap; overflow: hidden; text-overflow: ellipsis; opacity: .88;}
.bar-track {display: block; height: 9px; border-radius: 999px; background: rgba(148,163,184,.20); overflow: hidden;}
.bar-fill {display: block; height: 100%; border-radius: 999px; transition: width .55s ease;}
.bar-val {text-align: right; opacity: .78; font-variant-numeric: tabular-nums;}
.note {
  margin-top: 1.05rem; font-size: .79rem; line-height: 1.45;
  padding: .6rem .8rem; border-radius: 12px; color: #fbbf24;
  background: rgba(245,158,11,.13); border: 1px solid rgba(245,158,11,.35);
}

/* ---------- steps ---------- */
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .7rem; margin-top: .35rem;}
.step {
  border-radius: 16px; padding: .9rem 1rem;
  background: rgba(148,163,184,.07); border: 1px dashed rgba(148,163,184,.30);
}
.step b {font-size: .88rem;}
.step p {margin: .28rem 0 0; font-size: .78rem; line-height: 1.45; opacity: .68;}

/* ---------- empty / error card ---------- */
.empty {
  border-radius: 22px; padding: 1.5rem 1.6rem;
  background: rgba(148,163,184,.07); border: 1px solid rgba(148,163,184,.22);
}
.empty h3 {margin: 0 0 .35rem; font-size: 1.05rem;}
.empty p {margin: 0; font-size: .85rem; line-height: 1.55; opacity: .72;}
.foot {font-size: .74rem; opacity: .5; margin-top: 1.6rem; text-align: center; line-height: 1.6;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the network ...")
def get_model(checkpoint_path: str, device_name: str):
    """Load (and cache) the trained network."""
    return load_model_for_inference(checkpoint_path, device=device_name)


def hero_html(class_names: list[str]) -> str:
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:{style_for(n)["color"]}"></span>'
        f"{html.escape(n)}</span>"
        for n in class_names
    )
    return (
        '<div class="hero">'
        '<span class="hero-badge">ResNet-18 · trained from scratch</span>'
        "<h1>Satellite Disaster <span>Classifier</span></h1>"
        "<p>Drop in an aerial or satellite frame and the network reads the scene: "
        "earthquake damage, flooding, fire and smoke — or nothing unusual at all.</p>"
        f'<div class="chips">{chips}</div>'
        "</div>"
    )


def stats_html(metadata: dict, model: torch.nn.Module, img_size: int, num_classes: int) -> str:
    metrics = metadata.get("metrics") or {}
    cards: list[tuple[str, str]] = []
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    if accuracy is not None:
        cards.append((f"{accuracy * 100:.2f}%", "validation accuracy"))
    if macro_f1 is not None:
        cards.append((f"{macro_f1 * 100:.2f}%", "macro F1"))
    cards.append((f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", "parameters"))
    cards.append((f"{img_size} px", "input size"))
    cards.append((str(num_classes), "classes"))
    body = "".join(
        f'<div class="stat"><b>{value}</b><span>{label}</span></div>'
        for value, label in cards
    )
    return f'<div class="stats">{body}</div>'


def result_html(result: dict, class_names: list[str]) -> str:
    label = result["label"]
    confidence = result["confidence"]
    style = style_for(label)

    ranked = sorted(class_names, key=lambda name: result["probabilities"][name], reverse=True)
    bars = "".join(
        '<div class="bar-row">'
        f'<span class="bar-name">{html.escape(name)}</span>'
        '<span class="bar-track">'
        f'<span class="bar-fill" style="width:{result["probabilities"][name] * 100:.1f}%;'
        f'background:{style_for(name)["color"]}"></span>'
        "</span>"
        f'<span class="bar-val">{result["probabilities"][name] * 100:.1f}%</span>'
        "</div>"
        for name in ranked
    )

    runner_up = ranked[1] if len(ranked) > 1 else None
    runner_text = (
        f'Next best: {html.escape(runner_up)} · {result["probabilities"][runner_up] * 100:.1f}%'
        if runner_up
        else ""
    )

    note = ""
    if confidence < 0.55:
        note = (
            '<div class="note">Low confidence — this frame may not look like the aerial '
            "imagery the model was trained on (a map, a ground-level photo or heavy cloud "
            "will do that).</div>"
        )

    return (
        '<div class="spotlight">'
        '<div class="verdict">'
        f'<div class="ring" style="--p:{confidence * 100:.1f};--c:{style["color"]}">'
        '<div class="ring-inner">'
        f"<b>{confidence * 100:.0f}%</b><small>confidence</small>"
        "</div></div>"
        "<div>"
        '<div class="verdict-label">Most likely scene</div>'
        f'<div class="verdict-class">{style["emoji"]} {html.escape(label)}</div>'
        f'<div class="verdict-sub">{html.escape(style["blurb"])}</div>'
        f'<div class="verdict-sub" style="margin-top:.35rem;opacity:.55">{runner_text}</div>'
        "</div></div>"
        f'<div class="bars">{bars}</div>'
        f"{note}"
        "</div>"
    )


def main() -> None:
    st.markdown(
        hero_html(["Earthquake", "Flood", "Fire", "Normal"]), unsafe_allow_html=True
    )

    # ---- settings: collapsed, out of the way ---------------------------------
    with st.expander("⚙️ Advanced settings"):
        left, mid, right = st.columns([2.2, 1, 1])
        checkpoint_input = left.text_input(
            "Checkpoint file",
            value=str(DEFAULT_CHECKPOINT),
            help="Path to a best.pt produced by train.py",
        )
        has_cuda = torch.cuda.is_available()
        device_name = mid.selectbox(
            "Device",
            options=["cuda", "cpu"],
            index=0 if has_cuda else 1,
            format_func=lambda value: "GPU (CUDA)" if value == "cuda" else "CPU",
            help="Your RTX 3050 shows up as GPU (CUDA).",
        )
        use_tta = right.checkbox(
            "Test-time augmentation",
            value=True,
            help="Average the prediction with a mirrored copy of the image. "
            "Slightly more accurate, slightly slower.",
        )

    checkpoint_path = Path(checkpoint_input).expanduser()
    if not checkpoint_path.exists():
        st.markdown(
            '<div class="empty"><h3>No trained model yet</h3>'
            "<p>The app needs a checkpoint before it can classify anything. "
            "Download the data and train once (about 30–60 minutes on an RTX 3050):</p></div>",
            unsafe_allow_html=True,
        )
        st.code(
            "python download_data.py\n"
            "python train.py --data-dir data/aider --arch resnet18 --epochs 30",
            language="bash",
        )
        st.caption("Already trained? Point *Advanced settings → Checkpoint file* at your `best.pt`.")
        st.stop()

    try:
        model, class_names, img_size, metadata = get_model(
            str(checkpoint_path), device_name
        )
    except Exception as error:  # noqa: BLE001 - show the real message
        st.error(f"Could not load the checkpoint:\n\n`{error}`")
        st.stop()

    st.markdown(
        stats_html(metadata, model, img_size, len(class_names)), unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload an aerial image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            '<div class="section-title" style="margin-top:1.1rem">How it works</div>'
            '<div class="steps">'
            '<div class="step"><b>1 · Upload</b><p>Satellite, drone or aerial frame — '
            "JPG, PNG, BMP, WEBP or TIFF.</p></div>"
            '<div class="step"><b>2 · Inference</b><p>224 px centre crop, ResNet-18, '
            "one forward pass with test-time augmentation.</p></div>"
            '<div class="step"><b>3 · Read the result</b><p>Class, confidence and the '
            "full probability split across all four scenes.</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="foot">AIDERv2 aerial imagery · CC-BY-4.0 · '
            "educational model, not for live emergency decisions.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        image = Image.open(uploaded)
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be read as an image. Try a JPG or PNG.")
        return

    with st.spinner("Reading the scene ..."):
        result = predict_pil_image(
            model,
            image,
            class_names,
            img_size=img_size,
            device=device_name,
            tta=use_tta,
        )

    image_col, result_col = st.columns([1.05, 1], gap="large")

    with image_col:
        st.markdown('<div class="section-title">Input frame</div>', unsafe_allow_html=True)
        st.image(image)
        size_kb = len(uploaded.getvalue()) / 1024
        st.markdown(
            f'<div class="file-meta">{html.escape(uploaded.name)} · '
            f"{image.width}×{image.height} px · {size_kb:.0f} KB</div>",
            unsafe_allow_html=True,
        )

    with result_col:
        st.markdown('<div class="section-title">Model verdict</div>', unsafe_allow_html=True)
        st.markdown(result_html(result, class_names), unsafe_allow_html=True)

    st.markdown(
        '<div class="foot">AIDERv2 aerial imagery · CC-BY-4.0 · '
        "educational model, not for live emergency decisions.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
