import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from torchvision import transforms
import torch.nn.functional as F

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Plant Leaf Disease Detection", layout="centered")

# -----------------------
# LOAD MODELS
# -----------------------
@st.cache_resource
def load_models():
    # Classification Model
    class_model = torch.load("models/best_plantvillage_cnn.pth", map_location="cpu")
    class_model.eval()

    # Segmentation Model
    seg_model = tf.keras.models.load_model("models/Leaf_segmentation.h5")

    return class_model, seg_model

class_model, seg_model = load_models()

# -----------------------
# UI
# -----------------------
st.title("🌱 Plant Leaf Disease Detection & Segmentation")
st.write("Upload a leaf image to detect disease and visualize infected region.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

# -----------------------
# PREPROCESS
# -----------------------
def preprocess_classification(img):
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def preprocess_segmentation(img):
    img = img.resize((256,256))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------
# INFERENCE
# -----------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image", use_column_width=True)

    # Classification
    input_tensor = preprocess_classification(image)
    with torch.no_grad():
        outputs = class_model(input_tensor)
        pred = torch.argmax(outputs, 1).item()

    class_names = ["Healthy", "Blight", "Rust", "Gray Leaf Spot"]  # Edit based on your dataset
    predicted_class = class_names[pred] if pred < len(class_names) else "Unknown"

    st.success(f"🌿 Predicted Disease: {predicted_class}")

    # Segmentation
    seg_input = preprocess_segmentation(image)
    mask = seg_model.predict(seg_input)[0]

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.size[0], image.size[1]))

    st.image(mask, caption="Segmented Infected Region", use_column_width=True)
