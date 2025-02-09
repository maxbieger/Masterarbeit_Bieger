Code:
from ultralytics import YOLO

# 1. Modell laden (vortrainiertes YOLOv8-Modell)
model = YOLO("yolov8n.pt")  # yolov8m.pt = keine echtzeit aber genauer

DatensatzName= "Dataset_yolo"
image_dir = fr'C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\{DatensatzName}\Train'
data_dir = r'C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo'

# 2. Training starten
results = model.train(data=r"C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\data.yaml", epochs=100, imgsz=640)
# model.train(
#     data=r'C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\data.yaml',
#     epochs=5,                 # Anzahl der Epochen
#     imgsz=192,                 # Bildgröße
#     batch=16,                  # Batchgröße
#     workers=4                  # Threads
# )

# 3. Modell speichern
model.export(format="pt")  # Exportiere das trainierte Modell


results = model.val()
print(results)  # Zeigt Präzision, Recall und mAP an

# 1. Modell laden
model = YOLO("runs/detect/train/weights/best.pt")

# 2. Testbild analysieren
image_path = "dataset/images/test/dry_image.jpg"
results = model(image_path)

# 3. Ergebnisse visualisieren
results[0].save("annotated_image.jpg")  # Annotiertes Bild speichern
print('Done')

data.yaml:
path: C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo
train: Dataset_yolo\Train
val: Dataset_yolo\Validation
test: Dataset_yolo\Test
names:
    0: Healthy
    1: Dry


Datenstruktur:
Dataset_yolo/Train
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt

Fehler:
Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
train: Scanning C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\Dataset_yolo\Train.cache..
WARNING ⚠️ No labels found in C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\Dataset_yolo\
Train.cache, training may not work correctly. See https://docs.ultralytics.com/datasets/detect for dataset formatting guidance.val: Scanning C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\Dataset_yolo\Validation.cach
WARNING ⚠️ No labels found in C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\Yolo\Dataset_yolo\
Validation.cache, training may not work correctly. See https://docs.ultralytics.com/datasets/detect for dataset formatting guidance.
Plotting labels to C:\Users\maxbi\runs\detect\train13\labels.jpg... 
zero-size array to reduction operation maximum which has no identity

Model:
C:\Users\maxbi\runs\detect

Best: 50Epochen ab 2 unverändert
fitness: 0.9949989726027397
keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
maps: array([      0.995,       0.995])
names: {0: 'Healthy', 1: 'Dry'}
plot: True
results_dict: {'metrics/precision(B)': 0.9947481587762655, 'metrics/recall(B)': 0.9971802527874607, 'metrics/mAP50(B)': 0.995, 'metrics/mAP50-95(B)': 0.9949988584474886, 'fitness': 0.9949989726027397}
save_dir: WindowsPath('C:/Users/maxbi/runs/detect/train52')
speed: {'preprocess': 0.17320057839351888, 'inference': 71.2635365595476, 'loss': 0.0, 'postprocess': 0.47451055186851426}
task: 'detect'