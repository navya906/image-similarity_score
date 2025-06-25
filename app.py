import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

st.title("Image Similarity Search")
st.write("Upload two images to find their similarity.")

model= ResNet50(weights='imagenet', include_top=False, pooling='avg')
def preprocess_image(img):
    img=img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(img):
    img = preprocess_image(img)
    features = model.predict(img)
    return features

def compute_similarity(img1, img2):
    vec1 = extract_features(img1)
    vec2 = extract_features(img2)
    score = cosine_similarity(vec1, vec2)[0][0]
    return score

img1_file = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"], key="img1")
img2_file = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"], key="img2")

if img1_file and img2_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1_file, caption="Image 1", use_container_width=True)
    with col2:
        st.image(img2_file, caption="Image 2", use_container_width=True)

    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    similarity = compute_similarity(img1, img2)
    st.markdown(f"### 🔍 Similarity Score: `{similarity:.4f}`")




