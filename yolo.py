from ultralytics import YOLO
import os
import random
from PIL import Image
import albumentations as A
import cv2
import numpy as np

# ========================== KONFIGURATION ==========================
train = True
val = False

# Datensatz & Ausgabe-Verzeichnis anpassen
DatensatzName = "Dataset_yolo2"
dataset_path = fr'master\Yolo\{DatensatzName}'
output_dir = fr'master\Yolo\TrainingResults'  # Speichert Modell & Ergebnisse hier
val_output_dir = os.path.join(output_dir, "Validierung")  # Speichert Validierungsergebnisse
os.makedirs(output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Augmentierung aktivieren/deaktivieren
apply_augmentation = True

# ========================== AUGMENTIERUNG ==========================
def apply_augmentation_to_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konvertiere zu RGB

    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),  # Spiegelung
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),  # Randreflexion
        A.GaussNoise(var_limit=(5, 30), p=0.2)  # Leichtes Rauschen mit 20% Wahrscheinlichkeit
    ])

    augmented = augmentations(image=image)["image"]
    return augmented  # Gibt das augmentierte Bild zurück

# ========================== TRAINING ==========================
if train:
    # 1. YOLO-Modell laden (vortrainiertes Modell als Basis)
    model = YOLO("yolov8m.pt")  

    # 2. Training starten mit interner Augmentierung (Bilder werden NICHT gespeichert)
    model.train(
        data=os.path.join(dataset_path, "data.yaml"),  
        epochs=5,                 
        imgsz=192,                 
        batch=16,                  
        workers=4,                 
        save=True,
        project=output_dir,  # Speicherpfad für Ergebnisse
        name="yolo_experiment",
        augment=apply_augmentation  # Aktiviert Augmentierung nur während des Trainings
    )

    # 3. Modellvalidierung nach Training
    results = model.val()
    print("Training abgeschlossen. Ergebnisse:")
    print(results)

# ========================== VALIDIERUNG ==========================
if val:
    # 1. Modell aus dem gewählten Speicherpfad laden
    best_model_path = os.path.join(output_dir, "yolo_experiment", "weights", "best.pt")
    model = YOLO(best_model_path)

    # 2. Anzahl der Testbilder festlegen
    Anzahl_Bilder = 10
    test_image_dir = os.path.join(dataset_path, "Test")

    jpg_files = [f for f in os.listdir(test_image_dir) if f.endswith(".jpg")]

    for i in range(min(Anzahl_Bilder, len(jpg_files))):
        random_image = random.choice(jpg_files)
        image_path = os.path.join(test_image_dir, random_image)

        results = model(image_path, save=True, save_dir=val_output_dir)

        # Annotiertes Bild speichern
        annotated_image = results[0].plot()
        annotated_image = Image.fromarray(annotated_image)
        annotated_image.save(os.path.join(val_output_dir, f"annotated_image_{i}.jpg"))

        print(f"Bild {i+1} analysiert: {random_image}")

    # 3. Validierungsergebnisse ausgeben
    val_results = model.val(data=test_image_dir)
    print("Validierungsergebnisse:")
    print(f"mAP50: {val_results.mAP50}")
    print(f"mAP50-95: {val_results.mAP50_95}")
    print(val_results)

print("Prozess abgeschlossen.")
