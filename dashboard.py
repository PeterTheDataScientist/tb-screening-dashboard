"""
Clinical TB Screening Dashboard
Computer Vision and Image Analysis (HDS412) - Lab 2 Submission - Group 9

Simulates a clinical workstation for a radiologist screening chest X-rays
for tuberculosis. Provides AI-assisted prediction with three explainable AI
methods (Grad-CAM, LIME, SHAP).

Memory-optimised version for Streamlit Community Cloud (1 GB RAM limit).
Methods run sequentially with explicit garbage collection between them
so the container does not OOM when both LIME and SHAP are enabled.

USAGE:
    pip install streamlit torch torchvision opencv-python-headless pillow numpy
    pip install grad-cam shap lime scikit-image
    streamlit run dashboard.py
"""

import os
import io
import gc
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# XAI libraries
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="TB Screening AI - Clinical Decision Support",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .pred-banner {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .pred-tb { background-color: #ffe5e5; color: #c0392b; border: 3px solid #c0392b; }
    .pred-normal { background-color: #e5ffe5; color: #27ae60; border: 3px solid #27ae60; }
    .disclaimer {
        background-color: #fff3cd; color: #856404; padding: 0.75rem;
        border-radius: 5px; border-left: 4px solid #ffc107; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Memory-tuned XAI parameters (cut from original to fit 1 GB RAM)
LIME_NUM_SAMPLES = 200      # was 500
LIME_BATCH_SIZE = 16        # was 32
SHAP_NSAMPLES = 20          # was 50
SHAP_BG_SIZE = 4            # was 10


# ============================================================
# Memory utilities
# ============================================================
def free_memory():
    """Aggressively free memory between XAI methods."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Model
# ============================================================
def build_efficientnet_b0(num_classes=2):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def make_lung_oval_mask(size=224, vertical_stretch=1.1, horizontal_stretch=0.85):
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    a = (size / 2) * horizontal_stretch
    b = (size / 2) * vertical_stretch * 0.95
    oval = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2
    return (oval <= 1).astype(np.float32)


@st.cache_resource
def load_model(weights_path="best_model.pth"):
    if not Path(weights_path).exists():
        st.error(f"Model weights not found at {weights_path}.")
        st.stop()
    model = build_efficientnet_b0(num_classes=2)
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    return model, ckpt


@st.cache_resource
def get_lung_mask():
    return make_lung_oval_mask(IMG_SIZE)


# ============================================================
# Preprocessing + prediction
# ============================================================
def preprocess_image(pil_image, lung_mask):
    img_array = np.array(pil_image)
    if img_array.ndim == 2:
        img_gray = img_array
    elif img_array.ndim == 3 and img_array.shape[2] >= 3:
        img_gray = cv2.cvtColor(img_array[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array[..., 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_masked = (img_resized * lung_mask).astype(np.uint8)
    img_rgb = np.stack([img_masked, img_masked, img_masked], axis=-1)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    display_rgb = img_rgb.astype(np.float32) / 255.0
    return tensor, display_rgb, img_rgb


def predict(model, tensor, threshold=0.8081):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    prob_tb, prob_normal = float(probs[1]), float(probs[0])
    pred_class = "Tuberculosis" if prob_tb >= threshold else "Normal"
    if prob_tb < 0.20:
        risk = "LOW"
    elif prob_tb < threshold:
        risk = "MEDIUM"
    elif prob_tb < 0.95:
        risk = "HIGH"
    else:
        risk = "VERY HIGH"
    return {"pred_class": pred_class, "prob_tb": prob_tb,
            "prob_normal": prob_normal, "risk": risk, "threshold": threshold}


# ============================================================
# XAI methods (each one releases its own resources at the end)
# ============================================================
def compute_gradcam(model, tensor, display_rgb):
    target_layers = [model.features[-1]]
    cam_extractor = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(1)]
    cam = cam_extractor(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(display_rgb, cam, use_rgb=True, image_weight=0.5)
    del cam_extractor, cam
    free_memory()
    return overlay


def compute_lime(model, img_uint8, num_samples=LIME_NUM_SAMPLES):
    if not HAS_LIME:
        return None
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images_batch):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        batch = torch.stack([transform(img.astype(np.uint8)) for img in images_batch]).to(DEVICE)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        del batch
        return probs

    explanation = explainer.explain_instance(
        image=img_uint8,
        classifier_fn=predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        random_seed=42,
        batch_size=LIME_BATCH_SIZE,
    )
    temp, mask = explanation.get_image_and_mask(
        label=1, positive_only=True, num_features=10, hide_rest=False, min_weight=0.0
    )
    overlay = mark_boundaries(temp.astype(np.uint8) / 255.0, mask, color=(0, 1, 0), mode="thick")
    overlay = (overlay * 255).astype(np.uint8)
    del explainer, explanation, temp, mask
    free_memory()
    return overlay


def get_shap_background(lung_mask, n_bg=SHAP_BG_SIZE):
    """Build a small background tensor for SHAP from random noise.
    Not cached on purpose: re-creating is cheap and the cached tensor was
    holding ~50 MB resident across predictions."""
    bg = []
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    for _ in range(n_bg):
        noise = np.random.randint(50, 200, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        masked = (noise * lung_mask).astype(np.uint8)
        rgb = np.stack([masked]*3, axis=-1)
        bg.append(transform(rgb))
    return torch.stack(bg).to(DEVICE)


def compute_shap(model, tensor, display_rgb, lung_mask, nsamples=SHAP_NSAMPLES):
    if not HAS_SHAP:
        return None
    background = get_shap_background(lung_mask)
    explainer = shap.GradientExplainer(model, background, local_smoothing=0.5)
    shap_vals = explainer.shap_values(tensor, nsamples=nsamples)

    if isinstance(shap_vals, list):
        shap_tb = shap_vals[1][0]
    else:
        shap_tb = shap_vals[0, ..., 1]

    shap_2d = shap_tb.mean(axis=0) if shap_tb.ndim == 3 else shap_tb
    shap_abs = np.abs(shap_2d)
    shap_norm = shap_abs / shap_abs.max() if shap_abs.max() > 0 else shap_abs
    overlay = show_cam_on_image(display_rgb, shap_norm, use_rgb=True, image_weight=0.5)
    del explainer, shap_vals, background, shap_tb, shap_2d, shap_abs, shap_norm
    free_memory()
    return overlay


# ============================================================
# Main app
# ============================================================
def main():
    st.markdown('<div class="main-header">'
                '<h1 style="margin:0;">TB Screening AI - Clinical Decision Support</h1>'
                '<p style="margin:0;">EfficientNetB0 with Explainable AI (Grad-CAM, LIME, SHAP)</p>'
                '</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Patient Information")
        patient_id = st.text_input("Patient ID", value="ANON-001")
        exam_date = st.date_input("Exam Date", value=datetime.now())
        radiologist = st.text_input("Reviewing Clinician", value="Dr.")

        st.markdown("---")
        st.markdown("### AI Model Information")
        st.markdown("""
        **Architecture**: EfficientNetB0  
        **Training**: Lung-region masked  
        **Test AUC**: 0.9999  
        **Sensitivity**: 0.981  
        **Specificity**: 0.998  
        **Threshold**: 0.8081 (Youden's J)
        """)

        st.markdown("---")
        st.markdown("### XAI Settings")
        run_lime = st.checkbox("Enable LIME (slower)", value=True)
        run_shap = st.checkbox("Enable SHAP (slower)", value=True)
        st.caption(
            f"Cloud-tuned: LIME uses {LIME_NUM_SAMPLES} samples, "
            f"SHAP uses {SHAP_NSAMPLES} samples. "
            "Reduce further if the app crashes on free-tier hosting."
        )

        st.markdown("---")
        st.markdown('<div class="disclaimer">'
                    '<strong>DISCLAIMER:</strong> This system is a research prototype and '
                    'must NOT be used as a sole diagnostic instrument. All predictions '
                    'require qualified radiologist review.'
                    '</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### 1. Upload Chest X-ray")
        uploaded = st.file_uploader(
            "Choose a chest X-ray image",
            type=["png", "jpg", "jpeg"],
            help="PA-view chest X-ray, grayscale or RGB. Will be resized to 224x224."
        )
        st.markdown("**Or try a sample:**")
        samples_dir = Path("samples")
        if samples_dir.exists():
            sample_files = sorted([str(p) for p in samples_dir.glob("*.png")] +
                                    [str(p) for p in samples_dir.glob("*.jpg")])
            if sample_files:
                sample_choice = st.selectbox(
                    "Sample images",
                    options=["(none)"] + [Path(f).name for f in sample_files],
                )
                if sample_choice != "(none)":
                    uploaded = open(samples_dir / sample_choice, "rb")
        else:
            st.caption("No samples folder found. Upload a file above.")

    with col_right:
        if uploaded is None:
            st.info("Upload a chest X-ray to begin analysis.")
            st.markdown("""
            ### How this dashboard works
            1. Upload a PA-view chest X-ray image
            2. The AI processes the image (CLAHE enhancement, lung-region masking)
            3. The EfficientNetB0 model predicts Normal vs Tuberculosis
            4. Three explainable AI methods show what regions the model used:
               - **Grad-CAM**: gradient-weighted class activation maps
               - **LIME**: local interpretable model-agnostic explanations
               - **SHAP**: SHapley additive explanations
            5. Compare side-by-side to assess prediction reliability
            """)
            return

    pil_image = Image.open(uploaded).convert("RGB")
    model, ckpt = load_model("best_model.pth")
    lung_mask = get_lung_mask()
    tensor, display_rgb, img_uint8 = preprocess_image(pil_image, lung_mask)

    with st.spinner("Running AI analysis..."):
        prediction = predict(model, tensor, threshold=0.8081)

    pred = prediction["pred_class"]
    prob = prediction["prob_tb"]
    risk = prediction["risk"]
    banner_class = "pred-tb" if pred == "Tuberculosis" else "pred-normal"

    st.markdown(f'<div class="pred-banner {banner_class}">'
                f'PREDICTION: {pred}<br>'
                f'<span style="font-size: 1rem;">'
                f'TB probability: {prob:.1%} | Risk level: {risk} | '
                f'Threshold: {prediction["threshold"]:.4f}'
                f'</span></div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Normal probability", f"{prediction['prob_normal']:.1%}")
    col_b.metric("TB probability", f"{prediction['prob_tb']:.1%}")
    col_c.metric("Risk level", risk)

    st.markdown("---")
    st.markdown("### 2. AI Explanation (Compare Methods)")

    view_mode = st.radio(
        "View mode:",
        options=["All methods (side-by-side)", "Grad-CAM only", "LIME only", "SHAP only"],
        horizontal=True,
    )

    # ============================================================
    # Sequential, memory-aware XAI rendering
    # Each method computes -> renders -> frees its own memory.
    # On 1 GB Streamlit Cloud, running them in parallel killed the container.
    # ============================================================
    if view_mode == "All methods (side-by-side)":
        cols = st.columns(4)
        cols[0].image(display_rgb, caption="Original (preprocessed)", use_container_width=True)

        # 1. Grad-CAM
        with st.spinner("Computing Grad-CAM..."):
            try:
                gradcam_overlay = compute_gradcam(model, tensor, display_rgb)
                cols[1].image(gradcam_overlay, caption="Grad-CAM", use_container_width=True)
                del gradcam_overlay
                free_memory()
            except Exception as e:
                cols[1].error(f"Grad-CAM failed: {e}")

        # 2. LIME
        if run_lime and HAS_LIME:
            with st.spinner(f"Computing LIME ({LIME_NUM_SAMPLES} samples)..."):
                try:
                    lime_overlay = compute_lime(model, img_uint8)
                    if lime_overlay is not None:
                        cols[2].image(lime_overlay, caption="LIME", use_container_width=True)
                    del lime_overlay
                    free_memory()
                except Exception as e:
                    cols[2].error(f"LIME failed: {e}")
        else:
            cols[2].info("LIME disabled or not installed")

        # 3. SHAP
        if run_shap and HAS_SHAP:
            with st.spinner(f"Computing SHAP ({SHAP_NSAMPLES} samples)..."):
                try:
                    shap_overlay = compute_shap(model, tensor, display_rgb, lung_mask)
                    if shap_overlay is not None:
                        cols[3].image(shap_overlay, caption="SHAP", use_container_width=True)
                    del shap_overlay
                    free_memory()
                except Exception as e:
                    cols[3].error(f"SHAP failed: {e}")
        else:
            cols[3].info("SHAP disabled or not installed")

    elif view_mode == "Grad-CAM only":
        cols = st.columns(2)
        cols[0].image(display_rgb, caption="Original", use_container_width=True)
        with st.spinner("Computing Grad-CAM..."):
            gradcam_overlay = compute_gradcam(model, tensor, display_rgb)
            cols[1].image(gradcam_overlay, caption="Grad-CAM heatmap", use_container_width=True)

    elif view_mode == "LIME only":
        if run_lime and HAS_LIME:
            cols = st.columns(2)
            cols[0].image(display_rgb, caption="Original", use_container_width=True)
            with st.spinner("Computing LIME..."):
                lime_overlay = compute_lime(model, img_uint8)
                if lime_overlay is not None:
                    cols[1].image(lime_overlay, caption="LIME superpixels", use_container_width=True)
        else:
            st.info("Enable LIME in the sidebar to use this view.")

    elif view_mode == "SHAP only":
        if run_shap and HAS_SHAP:
            cols = st.columns(2)
            cols[0].image(display_rgb, caption="Original", use_container_width=True)
            with st.spinner("Computing SHAP..."):
                shap_overlay = compute_shap(model, tensor, display_rgb, lung_mask)
                if shap_overlay is not None:
                    cols[1].image(shap_overlay, caption="SHAP attribution", use_container_width=True)
        else:
            st.info("Enable SHAP in the sidebar to use this view.")

    # === How to interpret ===
    st.markdown("---")
    st.markdown("### 3. How to Interpret the Visualizations")

    with st.expander("Grad-CAM"):
        st.markdown("""
        Grad-CAM produces a coarse heatmap from the gradients flowing into the
        last convolutional layer. Red regions are where the model attended most
        strongly when classifying. **In our evaluation, Grad-CAM had the highest
        causal faithfulness (insertion/deletion AUC = +0.0736),** meaning its
        highlighted regions are most strongly tied to the model's actual decision.
        """)

    with st.expander("LIME"):
        st.markdown("""
        LIME divides the image into superpixels and randomly hides them to see
        which ones most affect the prediction. Green outlines mark the superpixels
        that support a TB diagnosis. **LIME produced visually plausible anatomical
        explanations but had negative causal faithfulness in our evaluation**, meaning
        its highlighted regions don't always drive the prediction. Use with caution
        for clinical decisions.
        """)

    with st.expander("SHAP"):
        st.markdown("""
        SHAP uses Shapley values from cooperative game theory to attribute
        prediction differences to individual pixels. Red regions support the TB
        prediction. **SHAP achieved the best balance of causal faithfulness
        (+0.0620) and clinical coherence (6.0/9 rubric score)** in our evaluation,
        and is our recommended method for clinical interpretation.
        """)

    with st.expander("Risk levels"):
        st.markdown("""
        - **LOW** (TB probability < 20%): No further action typically needed
        - **MEDIUM** (20% to threshold): Consider clinical correlation
        - **HIGH** (threshold to 95%): Recommend specialist review
        - **VERY HIGH** (> 95%): Urgent clinical review and confirmatory testing
        """)

    # === Generate report ===
    st.markdown("---")
    st.markdown("### 4. Generate Report")

    report_text = f"""TB SCREENING DIAGNOSTIC REPORT
================================
Date: {exam_date}
Patient ID: {patient_id}
Reviewing Clinician: {radiologist}

PREDICTION: {pred}
TB probability: {prob:.1%}
Risk level: {risk}
Decision threshold: {prediction['threshold']:.4f} (Youden's J)

Model: EfficientNetB0 (lung-region masked)
Test set performance: AUC = 0.9999, Sensitivity = 0.981, Specificity = 0.998

Recommended XAI: SHAP (combined faithfulness + clinical coherence)

DISCLAIMER: This is a research prototype. All predictions require
qualified radiologist review before clinical action.
"""
    st.download_button(
        "Download diagnostic report (.txt)",
        data=report_text,
        file_name=f"TB_report_{patient_id}_{exam_date}.txt",
        mime="text/plain",
    )

    # Final cleanup at end of run
    del tensor, display_rgb, img_uint8, pil_image
    free_memory()


if __name__ == "__main__":
    main()
