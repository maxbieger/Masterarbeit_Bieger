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
import cv2
import numpy as np
import random
from PIL import Image

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

# ========================== VALIDIERUNG ==========================
if val: #TODO not finished yet
    # Modell aus dem gewählten Speicherpfad laden
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
