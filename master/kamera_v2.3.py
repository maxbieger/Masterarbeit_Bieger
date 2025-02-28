import cv2
import time
from datetime import datetime
import os

def add_timestamp_to_image(image, timestamp):
    # Fügt die aktuelle Zeit als Text in die obere Ecke des Bildes ein.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2
    color = (255, 255, 255) 
    thickness = 1
    position = (5, 15)

    cv2.putText(image, timestamp, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def capture_images(interval_minutes, output_folder):
    # Nimmt Bilder in einem bestimmten Intervall auf und verwendet die Systemzeit.
    # Öffne die Kamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    print(f"Kamera erfolgreich geöffnet. Bilder werden alle {interval_minutes} Minuten aufgenommen.")

    try:
        while True:
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
            filename = f"{output_folder}/image_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"

            # Ein Frame von der Kamera aufnehmen
            ret, frame = cap.read()

            if not ret:
                print("Fehler: Bild konnte nicht aufgenommen werden.")
                break

            # In Graustufen umwandeln
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Anwenden einer Farbkarte (COLORMAP_JET)
            color_mapped_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)

            # Bild um 90 Grad drehen
            rotated_frame = cv2.rotate(color_mapped_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Zeit auf das transformierte Bild schreiben
            frame_with_timestamp = add_timestamp_to_image(rotated_frame, timestamp)

            # Bild speichern
            cv2.imwrite(filename, frame_with_timestamp)
            print(f"Bild gespeichert: {filename}")

            # Warten bis zum nächsten Intervall
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print("Aufnahme beendet.")
    finally:
        # Kamera freigeben
        cap.release()
        print("Kamera geschlossen.")

# Beispiel-Aufruf der Funktion
output_folder = "./captured_images/Unsorted_frames"  # Zielordner für die Bilder
interval_minutes = 5        # Intervall in Minuten

# Stelle sicher, dass der Zielordner existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

capture_images(interval_minutes, output_folder)
