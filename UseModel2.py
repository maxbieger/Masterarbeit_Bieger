from PIL import Image, ImageDraw
from keras.models import load_model
import numpy as np
import os
import random

# Pfade zu Bildern und Modell
image_path_1 = r'C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\captured_images\Anfangsphase mit wenig licht, wenig wasser'
model = load_model(r'C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Python Workspace\Jupyter\.ipynb_checkpoints\master\best_model20.01.2025_16-18-27.h5')

# Labels für Vorhersageklassen
labels = ['Healthy', 'Dry']

for i in range(30):
    # Zufälliges Bild auswählen
    image_files = os.listdir(image_path_1)
    random_image = random.choice(image_files)
    print(random_image)
    image_path = os.path.join(image_path_1, random_image)
    original_image = Image.open(image_path)

    # Originalgröße des Bilds speichern
    og_width, og_height = original_image.size

    # Bereich für das Modell vorbereiten
    new_width = int(og_width * 1.20)  # Skalierung auf 20 % der Originalgröße
    new_height = int(og_height * 1.20)

    resized_img = original_image.resize((new_width, new_height))
    #print('----Picture of a Healthy Plant: \n')

    # Manuelle Vorverarbeitung für das Modell
    preprocessed_image = original_image.resize((256, 192))
    preprocessed_image = np.array(preprocessed_image) / 255.0

    # Vorhersage durchführen
    preds = model.predict(np.expand_dims(preprocessed_image, axis=0))
    preds_class = np.argmax(preds)
    preds_label = labels[preds_class]

    print(f'Predicted Class: {preds_label}')
    print(f'Confidence Score: {preds[0][preds_class]}')

    # # Rechteck zeichnen auf das Originalbild
    # draw = ImageDraw.Draw(original_image)
    # rect_x1, rect_y1 = (og_width - 256) // 2, (og_height - 192) // 2
    # rect_x2, rect_y2 = rect_x1 + 256, rect_y1 + 192
    # draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], outline="red", width=5)

    # # Bild mit Rechteck anzeigen
    # original_image.show()
