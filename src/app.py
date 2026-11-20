"""Web-App: Bild per Webcam oder Upload klassifizieren."""

import streamlit as st
from PIL import Image

from classifier import load_model, load_class_names, predict_top_k


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_class_names():
    return load_class_names()


st.title("CIFAR-100 Bildklassifikator")
st.write(
    "Nimm ein Foto mit der Webcam auf oder lade ein Bild hoch. "
    "Ein selbst trainiertes CNN sagt vorher, welche der 100 CIFAR-100-Klassen "
    "darauf zu sehen ist."
)

# Bildquelle: Webcam oder Upload
camera_image = st.camera_input("Foto aufnehmen")
uploaded_file = st.file_uploader("... oder Bild hochladen", type=["png", "jpg", "jpeg"])

image_file = camera_image or uploaded_file

if image_file is not None:
    img = Image.open(image_file)
    st.image(img, caption="Dein Bild", width=224)

    results = predict_top_k(get_model(), get_class_names(), img, k=5)

    # Ergebnis anzeigen
    best_name, best_prob = results[0]
    st.success(f"Vorhersage: **{best_name}** ({best_prob * 100:.1f} %)")

    st.subheader("Top-5")
    for name, prob in results:
        st.write(f"{name}: {prob * 100:.1f} %")
        st.progress(prob)

    st.caption(
        "Hinweis: Das Modell erreicht ca. 58 % Accuracy auf CIFAR-100. "
        "Bei eigenen Fotos liegt es öfter daneben, denn es arbeitet intern mit nur 32 × 32 Pixeln."
    )
