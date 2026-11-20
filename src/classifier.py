"""Lädt das CIFAR-100-Modell und klassifiziert Bilder."""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # optional: blendet die oneDNN-Meldung aus

from tensorflow import keras
from tensorflow.image import resize

# Pfade relativ zu DIESER Datei, nicht zum Ordner, aus dem man das Skript startet
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "webcam_classifier.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"


def load_model():
    """Lädt das gespeicherte Keras-Modell und gibt es zurück."""
    model = keras.models.load_model(MODEL_PATH)

    return model 


def load_class_names():
    """Liest die Klassennamen aus der JSON-Datei und gibt sie als Liste zurück."""
    
    with open(CLASS_NAMES_PATH) as f:

        loaded = json.load(f)
    
    return loaded


def preprocess(img):
    """Bringt ein PIL-Bild in die Form, die das Modell erwartet.

    Rückgabe: numpy-Array mit Shape (1, 32, 32, 3) und Werten zwischen 0 und 1.
    """
    

    img = img.convert("RGB")

    img_array = resize(np.array(img), [32, 32])

    # Convert to numpy array and normalize pixel values between 0 and 1
    img_array = np.array(img_array).astype("float32") / 255.0 

    # Add batch dimension for prediction
    return img_array.reshape(1, 32, 32, 3)



def predict_top_k(model, class_names, img, k=5):
    """Gibt die k wahrscheinlichsten Klassen zurück.

    Rückgabe: Liste von (klassenname, wahrscheinlichkeit), beste zuerst,
    z. B. [("cup", 0.42), ("bowl", 0.17), ...]
    """
    # Wahrscheinlichkeiten für das eine Bild (Shape (100,) statt (1, 100))

    x = preprocess(img)
    probs = model.predict(x, verbose = 0)[0]



    # Indizes der 5 höchsten Wahrscheinlichkeiten, beste zuerst
    top_indices = np.argsort(probs)[-k:][::-1]

    results = []

    for i in top_indices:

       results.append((class_names[i], float(probs[i])))       
    return results


if __name__ == "__main__":
    model = load_model()
    class_names = load_class_names()

    test_image = PROJECT_DIR / "imageclassifier" / "data" / "captures" / "15-15-42-901409.png"
    img = Image.open(test_image)

    for rank, (name, prob) in enumerate(predict_top_k(model, class_names, img), start=1):
        print(f"{rank}. {name:<15} {prob * 100:5.1f} %")

