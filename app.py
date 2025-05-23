import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_model_final.keras', compile=False)

model = load_model()

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24

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

st.title("Handwriting Recognition with Keras Model")

st.write("""
Upload a grayscale or color image of handwritten text.
The model expects images preprocessed to 256x64
""")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict Text"):
        input_img = preprocess(img)
        pred = model.predict(input_img)
        predicted_text = decode_prediction(pred)
        if len(predicted_text) > 2:
            predicted_text = predicted_text[:]
        st.markdown("### Predicted Text:")
        st.write(predicted_text if predicted_text.strip() else "(Nothing detected)")
else:
    st.warning("Please upload an image first.")
