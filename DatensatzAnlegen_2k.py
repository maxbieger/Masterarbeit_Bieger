import os
import shutil
from itertools import cycle
import time

#--------------------Variable Eingabe--------------------------

DatensatzName= "Dataset_Restnet_mini"

# Variablen mit Pfaden zu den Eingabe- und Ausgangsordnern
#Dry
eingabe_ordner1 = r"C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\captured_images\unter 30 prozent feuchtigkeit"
#SuperDry
eingabe_ordner2 = r"C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\captured_images\unter 15 prozent feuchtigkeit"
#Healty
eingabe_ordner3 =  r"C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\captured_images\normal"

# Wo soll Datensatz erstellt werden
ausgangs_ordner_basis = r"C:\Users\maxbi\OneDrive\Dokumente\Masterstudiengang\Masterarbeit\Gültas\master\ResNet"#captured_images"

# Verteilung in Prozent (z.B. 30%, 50%, 20%)
#Wichtig genug validation mind 30%
verteilung = [0.65, 0.05, 0.3] #Test, Train, Validation
MaxAbweichungGesamt = 0.02
Verteilung_durch_duplizieren_ausgleichen = False #TODO Passe noch die Befehle in Zeile 155 an, im augenblick wird Healthy an Dry angepasst
# Wichtig: Die Prozentpunkte enspricht der batch größe welche kopiert wird, bei unter 100 Bilder erhält Validation nur den Rest

#--------------------------------------------------------------

# Zielordner definieren
ausgabe_test_dry = os.path.join(ausgangs_ordner_basis, DatensatzName, "Test\Dry")
ausgabe_train_dry = os.path.join(ausgangs_ordner_basis, DatensatzName, "Train\Dry")
ausgabe_validation_dry = os.path.join(ausgangs_ordner_basis, DatensatzName, "Validation\Dry")

# ausgabe_test_superdry = os.path.join(ausgangs_ordner_basis, DatensatzName, "Test/SuperDry")
# ausgabe_train_superdry = os.path.join(ausgangs_ordner_basis, DatensatzName, "Train/SuperDry")
# ausgabe_validation_superdry = os.path.join(ausgangs_ordner_basis, DatensatzName, "Validation/SuperDry")

ausgabe_test_healthy = os.path.join(ausgangs_ordner_basis, DatensatzName, "Test\Healthy")
ausgabe_train_healthy = os.path.join(ausgangs_ordner_basis, DatensatzName, "Train\Healthy")
ausgabe_validation_healthy = os.path.join(ausgangs_ordner_basis, DatensatzName, "Validation\Healthy")

def sicherstellen_ordnerstruktur(basis_pfad):
    struktur = [
        "Test\Dry",  "Test\Healthy",
        "Train\Dry",  "Train\Healthy",
        "Validation\Dry", "Validation\Healthy"
    ]
    
    basis_pfad = os.path.join(basis_pfad, DatensatzName)
    
    for pfad in struktur:
        kompletter_pfad = os.path.join(basis_pfad, pfad)
        if not os.path.exists(kompletter_pfad):
            os.makedirs(kompletter_pfad)

# Ordnerstruktur anlegen
sicherstellen_ordnerstruktur(ausgangs_ordner_basis)

# Überprüfen, ob die Summe der Verteilung 1.0 ergibt
if sum(verteilung) != 1.0:
    raise ValueError("Die Verteilungssummen müssen gleich 1.0 sein!")

# Eingabeordner und zugehörige Zielordner definieren
eingabe_zu_ausgabe = {
    eingabe_ordner1: [ausgabe_test_dry, ausgabe_train_dry, ausgabe_validation_dry],
    eingabe_ordner2: [ausgabe_test_dry, ausgabe_train_dry, ausgabe_validation_dry],
    eingabe_ordner3: [ausgabe_test_healthy, ausgabe_train_healthy, ausgabe_validation_healthy]
}

# Dateien aufteilen
for eingabe_ordner, ausgabe_ordner_liste in eingabe_zu_ausgabe.items():
    bilder = [os.path.join(eingabe_ordner, datei) for datei in os.listdir(eingabe_ordner) if os.path.isfile(os.path.join(eingabe_ordner, datei))]
    bilder.sort()  # Sortieren für konsistente Verteilung
    
    zyklus = []
    for idx, prozent in enumerate(verteilung):
        zyklus.extend([ausgabe_ordner_liste[idx]] * int(prozent * 100))  # Skalieren für verlässlichen Zyklus
    zyklus = cycle(zyklus)

    for bild in bilder:
        ziel_ordner = next(zyklus)  # Nächsten Ausgangsordner im Zyklus auswählen
        shutil.copy(bild, os.path.join(ziel_ordner, os.path.basename(bild)))

print(f"{DatensatzName}: Bilder verteilt!")



# Testbereich: Verteilung überprüfen und ausgeben
def berechne_verteilung(ordner_liste):
    gesamt_anzahl = 0
    verteilung_dict = {}

    for ordner in ordner_liste:
        anzahl_dateien = len([datei for datei in os.listdir(ordner) if os.path.isfile(os.path.join(ordner, datei))])
        verteilung_dict[ordner] = anzahl_dateien
        gesamt_anzahl += anzahl_dateien

    #print("Verteilung in den Zielordnern:")
    gesamt_kategorien = {"Test": 0, "Train": 0, "Validation": 0}

    for ordner, anzahl in verteilung_dict.items():
        prozent = (anzahl / gesamt_anzahl) * 100 if gesamt_anzahl > 0 else 0
        #print(f"{ordner}: {anzahl} Dateien ({prozent:.2f}%)")
        if "Test" in ordner:
            gesamt_kategorien["Test"] += anzahl
        elif "Train" in ordner:
            gesamt_kategorien["Train"] += anzahl
        elif "Validation" in ordner:
            gesamt_kategorien["Validation"] += anzahl

    print("\nGesamtverteilung nach Kategorien:")
    gesamt_anzahl_kategorien = sum(gesamt_kategorien.values())
    korrekt = True
    for kategorie, anzahl in gesamt_kategorien.items():
        prozent = (anzahl / gesamt_anzahl_kategorien) * 100 if gesamt_anzahl_kategorien > 0 else 0
        target_prozent = verteilung[list(gesamt_kategorien.keys()).index(kategorie)] * 100
        print(f"{kategorie}: {anzahl} Dateien ({prozent:.2f}%)")
        if abs(prozent - target_prozent) > (MaxAbweichungGesamt*100):
            korrekt = False

    if korrekt:
        print("\n"+"Verteilung korrekt, Abweichung eingehalten.")
    else:
        print("\n"+"Verteilung nicht korrekt, Abweichung nicht eingehalten.")

alle_zielordner = [
    ausgabe_test_dry, ausgabe_train_dry, ausgabe_validation_dry,
    #ausgabe_test_superdry, ausgabe_train_superdry, ausgabe_validation_superdry,
    ausgabe_test_healthy, ausgabe_train_healthy, ausgabe_validation_healthy
]
berechne_verteilung(alle_zielordner)
if Verteilung_durch_duplizieren_ausgleichen:
    def count_images_in_path(path):
        # Liste aller Dateien im angegebenen Verzeichnis abrufen
        bilder = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        # Rückgabe der Anzahl der Bilder
        return len(bilder)
    def duplizieren_bilder(pfad, target_anzahl):
        bilder = [f for f in os.listdir(pfad) if os.path.isfile(os.path.join(pfad, f))]
        aktuelle_anzahl = len(bilder)
        if aktuelle_anzahl < target_anzahl:
            fehlende_bilder = target_anzahl - aktuelle_anzahl
        
            
            # Duplizieren der Bilder
            bilder_zu_duplizieren = bilder[:fehlende_bilder]
            
            for bild in bilder_zu_duplizieren:
                quelle = os.path.join(pfad, bild)
                ziel = os.path.join(pfad, f"copy_of_{bild}")
                shutil.copy2(quelle, ziel)
                

    # Überprüfung und Duplizieren für jeden Pfad
    duplizieren_bilder(ausgabe_test_healthy,count_images_in_path(ausgabe_test_dry))
    duplizieren_bilder(ausgabe_train_healthy,count_images_in_path(ausgabe_train_dry))
    duplizieren_bilder(ausgabe_validation_healthy,count_images_in_path(ausgabe_validation_dry))

    print("Überprüfung und Verdopplung abgeschlossen.")