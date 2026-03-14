import streamlit as st
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torchvision
from anomalib.models import Padim
from anomalib.engine import Engine
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tempfile


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(page_title="Satellite Anomaly Detection", layout="wide")

st.title("🌍 Satellite Anomaly Detection System")
st.write("Upload a satellite image to detect anomalies and classify them.")


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------
# PATHS
# --------------------------------------------------

classifier_path = Path("/Users/tejasy/Desktop/JARVIS/classifier.pth")

ckpt_files = list(Path("/Users/tejasy/Desktop/hack").rglob("*.ckpt"))

if len(ckpt_files) == 0:
    st.error("No PaDiM checkpoint found.")
    st.stop()

padim_ckpt = ckpt_files[0]


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "Deforestation",
    "Illegal Building",
    "River Encroachment"
]


# --------------------------------------------------
# LOAD MODELS (cached)
# --------------------------------------------------

@st.cache_resource
def load_models():

    class Classifier(torch.nn.Module):

        def __init__(self, output_shape):
            super().__init__()

            weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            self.model = torchvision.models.efficientnet_b0(weights=weights)

            for param in self.model.features.parameters():
                param.requires_grad = False

            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(1280, output_shape)
            )

        def forward(self, x):
            return self.model(x)

    classifier = Classifier(len(class_names)).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    padim_model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"]
    )

    engine = Engine(
        accelerator="cpu",
        default_root_dir="/Users/tejasy/Desktop/hack/results"
    )

    return classifier, padim_model, engine


classifier, padim_model, engine = load_models()


# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(img, use_container_width=True)

    input_tensor = transform(img).unsqueeze(0).to(device)


    # --------------------------------------------------
    # TEMP FILE FIX
    # --------------------------------------------------

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        temp_path = tmp.name


    # --------------------------------------------------
    # RUN ANOMALY DETECTION
    # --------------------------------------------------

    with st.spinner("Running anomaly detection..."):

        predictions = engine.predict(
            model=padim_model,
            data_path=temp_path,
            ckpt_path=str(padim_ckpt)
        )

        result = predictions[0]

        anomaly_score = result.pred_score.item()
        anomaly_label = result.pred_label.item()


    st.metric("Anomaly Score", round(anomaly_score,4))


    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------

    if anomaly_label:

        with torch.inference_mode():
            preds = classifier(input_tensor)

        pred_class = torch.argmax(preds, dim=1).item()

        label = class_names[pred_class]

        st.success(f"Detected Anomaly: **{label}**")

    else:

        label = "Normal Region"

        st.success("No anomaly detected")


    # --------------------------------------------------
    # ANOMALY MAP
    # --------------------------------------------------

    anomaly_map = result.anomaly_map.squeeze().cpu().numpy()

    anomaly_map = (
        anomaly_map - anomaly_map.min()
    ) / (
        anomaly_map.max() - anomaly_map.min()
    )

    img_np = np.array(img) / 255.0

    h, w, _ = img_np.shape

    anomaly_map = cv2.resize(anomaly_map, (w, h))

    heatmap = plt.cm.jet(anomaly_map)[:, :, :3]

    overlay = 0.6 * img_np + 0.4 * heatmap


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img_np)

    with col2:
        st.subheader("Anomaly Overlay")
        st.image(overlay)


    st.markdown(
        """
        ### Color Guide

        🔵 **Blue** → Normal Region  
        🟡 **Yellow** → Moderate Anomaly  
        🔴 **Red** → Strong Anomaly
        """
    )