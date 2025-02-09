from ultralytics import YOLO
import os
import random
from PIL import Image

train = True
val=True
DatensatzName = "Dataset_yolo2"
image_dir = fr'C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\{DatensatzName}\Train'
data_dir = r'C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo'

if train:
    # 1. Modell laden (vortrainiertes YOLOv8-Modell)
    model = YOLO("yolov8m.pt")  # yolov8m.pt = keine echtzeit aber genauer

    # 2. Training starten
    #results = model.train(data=r"C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\data.yaml", epochs=100, imgsz=640)
    model.train(
        data=r'C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\data.yaml',
        epochs=50,                 # Anzahl der Epochen
        imgsz=192,                 # Bildgröße
        batch=16,                  # Batchgröße
        workers=4                  # Threads
    )

    results = model.val()
    #print(results)  # Zeigt Präzision, Recall und mAP an

    # 4. Genauigkeit ausgeben
    print(results)
    # print(results.maps[])  # Access mAP for IOU=0.5
    # print(results.mAP50_95)  # Access mAP for IOU=0.5:0.95


if val:
    # 1. Modell laden
    model = YOLO(r"C:\Users\maxbi\runs\detect\train\weights\best.pt")

    for i in range(10):
        # 2. Testbild analysieren
        #image_path = os.path.join(data_dir,DatensatzName, "Test")
        image_path = r"C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\Dataset_yolo2\Test"
        jpg_files = [file for file in os.listdir(image_path) if file.endswith('.jpg')]
        random_image = random.choice(jpg_files)
        results = model(os.path.join(image_path, random_image), save=True, save_dir=data_dir)

        # Access the annotated image as a NumPy array
        annotated_image = results[0].plot()  # Returns the image with annotations
        annotated_image = Image.fromarray(annotated_image)  # Convert to a PIL Image

        # Save it to a file
        annotated_image.save(f"annotated_image_{i}.jpg")


        
        # 3. Validierung des Modells auf dem Testdatensatz
        results = model.val(data=image_path)  # Testdatensatz muss in der .yaml-Datei definiert sein

        # 4. Ergebnisse ausgeben
        print("Validierungsergebnisse:")
        print(f"mAP50: {results.mAP50}")  # Mean Average Precision bei IOU=0.5
        print(f"mAP50-95: {results.mAP50_95}")  # Mean Average Precision bei IOU=0.5:0.95
        print(results)

    print('Done')
    ###############

    

    # val_dir = os.path.join(data_dir, DatensatzName, "Test")  # Verzeichnis mit Validierungsbildern

    # # 1. Modell laden
    # model = YOLO(r"C:\Users\maxbi\runs\detect\train15\weights\best.pt")

    # # 2. Alle Bilder im Validierungsverzeichnis analysieren
    # jpg_files = [file for file in os.listdir(val_dir) if file.endswith('.jpg')]
    # total_images = len(jpg_files)

    # if total_images == 0:
    #     print("Keine Bilder im Validierungsverzeichnis gefunden.")
    # else:
    #     correct_predictions = 0

    #     for img_file in jpg_files:
    #         # Bildpfad
    #         img_path = os.path.join(val_dir, img_file)

    #         # Vorhersage durchführen
    #         results = model(img_path)

    #         # Ergebnisse auswerten
    #         for result in results:
    #             # Hier kannst du die Logik implementieren, um die Korrektheit zu bestimmen.
    #             # Beispiel: Anzahl der erkannten Objekte oder Vergleich mit Ground Truth.
    #             # In diesem Beispiel wird eine zufällige Bedingung verwendet:
    #             if len(result.boxes) > 0:  # Annahme: Ein korrektes Bild hat mindestens eine erkannte Box
    #                 correct_predictions += 1

    #     # 3. Genauigkeit berechnen
    #     accuracy = (correct_predictions / total_images) * 100
    #     print(f"Validierungs-Genauigkeit: {accuracy:.2f}%")
