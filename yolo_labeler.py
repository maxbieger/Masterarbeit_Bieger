import os
from PIL import Image  # Importieren von Image aus der PIL-Bibliothek
import shutil

# Basisverzeichnis des Datensatzes
Dataset_name = 'Dataset_yolo2'
base_dir = fr'master\Yolo\{Dataset_name}'


#Aufgaben die dieses Programm erfüllt
Bildnamen_bereinigen = False #Falsche namen entfernen
labels_erstellen=False #labels erstellen
Ordner_Struktur_bereinigen=True #struktur anpassen

# Bildnamen bereinigen
if Bildnamen_bereinigen:
    splits = ["Train", "Validation", "Test"]
    klasses = ["Healthy", "Dry"]

    for split in splits:
        for k in klasses:
            image_dir = os.path.join(base_dir, split,k)
            #print(image_dir)
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    # Entferne ungültige Zeichen aus den Bildnamen
                    valid_image_name = ''.join(c for c in file if c.isalnum() or c in ['.', '_', ' ']).strip()

                    if valid_image_name:
                        # Erstelle den neuen Pfad mit bereinigtem Bildnamen
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(root, valid_image_name)

                        # Wenn sich der Name geändert hat, kopiere oder benenne das Bild um
                        if old_path != new_path:
                            os.rename(old_path, new_path)

    print("Bildnamen wurden bereinigt!")

if labels_erstellen:
    # Splits
    splits = ["Train", "Validation", "Test"]
    klasses = ["Healthy", "Dry"]

    name_to_number = {
        "Healthy": 0,
        "Dry": 1
    }
    # Labels generieren
    for split in splits:
        for k in klasses:
            image_dir = os.path.join(base_dir, split,k)
            print(image_dir)
            label_dir = os.path.join(base_dir, split,k)
            os.makedirs(label_dir, exist_ok=True)

            for image_name in os.listdir(image_dir):
                if image_name.endswith((".jpg", ".png")):
                    image_path = os.path.join(image_dir, image_name)
                    image = Image.open(image_path)
                    width, height = image.size

                    # YOLO-Label erstellen
                    label_file = os.path.join(label_dir, f"{image_name.split('.')[0]}.txt")
                    with open(label_file, "w") as f:
                        x_center = 0.5  # zentriert
                        y_center = 0.5  # zentriert
                        rel_width = 1.0  # volle Breite
                        rel_height = 1.0  # volle Höhe

                        f.write(f"{name_to_number[k]} {x_center} {y_center} {rel_width} {rel_height}\n")
    print("Labels wurden erfolgreich erstellt!")

if Ordner_Struktur_bereinigen:
    def unpack_folder(folder_path):
        # Überprüfen, ob der angegebene Pfad ein Ordner ist
        if not os.path.isdir(folder_path):
            print(f"Der Pfad '{folder_path}' ist kein gültiger Ordner.")
            return

        # Ermitteln des übergeordneten Verzeichnisses
        parent_directory = os.path.dirname(folder_path)

        # Dateien und Unterverzeichnisse im Ordner iterieren
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # Verschieben von Dateien und Verzeichnissen in das übergeordnete Verzeichnis
            new_path = os.path.join(parent_directory, item)
            shutil.move(item_path, new_path)
        # Löschen des jetzt leeren Ordners
        os.rmdir(folder_path)

    # Splits
    splits = ["Train", "Validation", "Test"]
    klasses = ["Healthy", "Dry"]

    # Labels generieren
    for split in splits:
        for k in klasses:
            image_dir = os.path.join(base_dir, split,k)
            print(image_dir)
            unpack_folder(image_dir)

    print("Ordnerstruktur wurde bereinigt!")
