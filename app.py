import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import base64
import io

# Constants
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24

st.title("✍️ Handwriting Recognition Model with CNN, RNN & Bi-LSTM")
st.write("Download a sample handwritten image below or upload your own.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model_final.keras", compile=False)

model = load_model()

def preprocess(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h, w) = img.shape
    final_img = np.ones([64, 256], dtype=np.float32) * 255
    if w > 256:
        img = img[:, :256]
        w = 256
    if h > 64:
        img = img[:64, :]
        h = 64
    final_img[:h, :w] = img
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    final_img = final_img / 255.0
    return final_img.reshape(1, 256, 64, 1)

def decode_prediction(pred):
    pred_indices = np.argmax(pred, axis=-1)[0]
    result = ""
    previous = -1
    blank_idx = len(alphabets)
    for idx in pred_indices:
        if idx == blank_idx or idx == previous:
            previous = idx
            continue
        if 0 <= idx < len(alphabets):
            result += alphabets[idx]
        previous = idx
    return result

def generate_download_link(image_path, label):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        b64 = base64.b64encode(image_bytes).decode()
        href = f'<a href="data:file/jpg;base64,{b64}" download="{label}">📥 Download</a>'
        return href

# --- SHOW SAMPLE IMAGES ---
st.subheader("📸 Sample Handwritten Images")

sample_images = {
    "Sample 1: Drawing board. Download and scribble over it to test": "board.jpg",
    "Sample 2": "kirtan.jpg",
    "Sample 3": "sample3.jpg"
}

for name, path in sample_images.items():
    try:
        img = Image.open(path).convert("L")
        st.markdown(generate_download_link(path, f"{name}.jpg"), unsafe_allow_html=True)
        st.image(img, caption=name, width=300)
        
    except FileNotFoundError:
        st.error(f"{name} not found: {path}")

# --- FILE UPLOAD & PREDICTION ---
st.subheader("Upload an image for prediction")
uploaded_file = st.file_uploader("Upload a handwritten image...", type=["png", "jpg", "jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(img, caption="Uploaded Image", use_column_width=True, channels="GRAY")

    if st.button("🔍 Predict Text"):
        input_img = preprocess(img)
        pred = model.predict(input_img)
        predicted_text = decode_prediction(pred)
        st.markdown("### 🧠 Predicted Text:")
        st.write(predicted_text if predicted_text.strip() else "(Nothing detected)")
