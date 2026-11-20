"""Klassifiziert ein Bild von der Kommandozeile aus.

Aufruf:  uv run python src/predict.py pfad/zum/bild.png --top 5
"""

import argparse
from pathlib import Path

from PIL import Image

from classifier import load_model, load_class_names, predict_top_k


def main():
    # 1. Kommandozeilen-Argumente definieren
    parser = argparse.ArgumentParser(description="Klassifiziert ein Bild mit dem CIFAR-100-Modell.")
    parser.add_argument("image", help="Pfad zum Bild")
    parser.add_argument("--top", type=int, default=5, help="Anzahl der angezeigten Klassen")
    args = parser.parse_args()

    # 2. Prüfen, ob die Datei existiert
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Datei nicht gefunden: {image_path}")
        return

    # 3. Bild öffnen, Modell und Klassennamen laden
    img = Image.open(image_path)
    model = load_model()
    class_names = load_class_names()

    # 4. Vorhersage
    results = predict_top_k(model, class_names, img, k= args.top)

    # 5. Ausgabe
    for rank, (name, prob) in enumerate(results, start=1):
            print(f"{rank}. {name:<15} {prob * 100:5.1f} %")


if __name__ == "__main__":
    main()
