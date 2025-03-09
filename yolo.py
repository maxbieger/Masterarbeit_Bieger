from ultralytics import YOLO
import os
import random
from PIL import Image
import cv2
import numpy as np

# ========================== KONFIGURATION ==========================
train = False
augmentation = True

# Datensatz & Ausgabe-Verzeichnis anpassen
DatensatzName = "Dataset_yolo2"
dataset_path = fr'master\Yolo\{DatensatzName}'
file_path = fr'master\Yolo'
output_dir = fr'master\Yolo\TrainingResults'  # Speichert Modell & Ergebnisse hier
val_output_dir = os.path.join(output_dir, "Validierung")  # Speichert Validierungsergebnisse
os.makedirs(output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Augmentierung aktivieren/deaktivieren
apply_augmentation = True

# ========================== AUGMENTIERUNG ==========================

def augment_image_opencv(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konvertiere zu RGB für PIL

    # Zufällige horizontale oder vertikale Spiegelung
    flip_type = random.choice([ -1]) #None,None,None, 1, 0, # 1 = horizontal, 0 = vertikal, -1 = beides
    if flip_type is not None:
        image = cv2.flip(image, flip_type)

    # Zufällige Rotation
    angle = random.uniform(-25, 25)  
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Zufälliges Rauschen hinzufügen
    if random.random() < 1:
        noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    return image

if augmentation:
    
    # Dateiname festlegen
    save_path = r'C:\Users\maxbi\GitHub\Masterarbeit_Bieger\master\Beispiel_Bild_mit_Augmentierung2.jpg'

    # Speichert das augmentierte Bild
    image = cv2.cvtColor(augment_image_opencv(r'C:\Users\maxbi\GitHub\Masterarbeit_Bieger\master\Beispiel_Bild_ohne_Augmentierung.jpg', save_path), cv2.COLOR_RGB2BGR) 
    cv2.imwrite(save_path, image)

# ========================== TRAINING ==========================
if train:
    # YOLO-Modell laden
    model = YOLO("yolov8m.pt")  

    # Training
    model.train(
        data=os.path.join(file_path, "data.yaml"),  
        epochs=5,                 
        imgsz=192,                 
        batch=16,                  
        workers=4,                 
        save=True,
        project=output_dir, 
        name="yolo_experiment",
        augment=apply_augmentation 
    )

    # validierung
    results = model.val()
    print("Training abgeschlossen. Ergebnisse:")
    print(results)
