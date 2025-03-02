from ultralytics import YOLO
import os
import random
from PIL import Image
import cv2
import numpy as np

# ========================== KONFIGURATION ==========================
train = True
val = False

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

def augment_image_opencv(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konvertiere zu RGB für PIL

    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    angle = random.uniform(-25, 25)  # Drehen 
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if random.random() < 0.2:
        noise = np.random.normal(0, 1, image.shape).astype(np.uint8)  # Leichtes Rauschen
        image = cv2.add(image, noise)

    return image

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
