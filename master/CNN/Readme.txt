Kamera 256x192
User: max Passwort: mB@Tum0066
Datei übersicht:
Kamera_v2.py = Ein foto speichern, aufsteigend nummerrieren, drehen, farb palette
Kamera_v2.1.py = dauerhafter Stream
Kamera_v2.2.py = jede sekunde aufnehmen und nicht mehr anzeigen
B_CNN.ipynb = Convolutional Neural Network 
Kamera_v2.3.py = Daueraufnahme sekunde aufnehmen und nicht mehr anzeigen, zeit syncro mit internet alle 5 bild jede minute
CNN.py = aktuelle arbeitsumgebenung
DatensatzAnlegen.py = nimmt die bilder aus den ordner und erzeugt daraus die datensatz struktur
CNN_2k.py = mit nur 2 klassen Dry und Healthy
DatensatzAnlegen_2k = erzeugt einen datensatz mit zwei klassen und teilt normal, 30%, 15% auf Dry und Healthy auf und kann bei bedarf die Bilder durch duplizierung vervielfältigen damit es gelich verteilt ist



How-To Enviroment:
1.python3 -m venv Env_CNN
2.source /home/max/master/Env_CNN/bin/activate
3.pip install tensorflow keras numpy matplotlib plotly scikit-learn pillow
(source Env1/bin/activate
oder source Env_CNN/bin/activate)


deactivate

Python prioriät auf 5 damit raspberry nicht crashed

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(None, 192, 256, 3), dtype=float32)
  • training=True
  • mask=None


Augmentation da die bilder alle aus einer richtung

Fragen:
- Tag oder nach aufnahmen = hätt eman überlegen können wenn wir schelchte werte hätten haben wir aber nciht
- Images resize = auf Pc genug power

2. Durchschnittliche Tendenzen
Learning Rate
learning_rate = 0.05 hat im Durchschnitt bessere Ergebnisse erzielt als 0.01 oder 0.2.
Bei 0.01: Solide Ergebnisse, aber keine Spitzenwerte.
Bei 0.05: Höchste Genauigkeit erreicht.
Bei 0.2: Meist instabil und führt oft zu schlechten Ergebnissen.
Layer-Konfiguration
Denselayer:
Denselayer = 3 hat durchgehend bessere Ergebnisse als Denselayer = 5.
layer1, layer2, layer3:
Kombinationen mit layer1 = 10, layer2 = 7 und layer3 = 3 waren besonders erfolgreich.
Zu wenig schichten haben zu einer accuracy festsetzung von 65% geführt, weil 64,8% Dry bilder waren un das programm einfach alles als dry ausgab

Es hat zwar etwas gedauert aber attention mechanism hat zur 91% accurcy geführt

Augmentation müsste eigentlich auf auf die validation ausgeführt werden oder?
Augmentierung wird verwendet, um das Trainingsset zu erweitern und das Modell robuster gegen Overfitting zu machen.
Test- und Validierungsdaten sollten nicht augmentiert werden, weil sie echte, unveränderte Daten simulieren sollen, die das Modell noch nicht gesehen hat.

Richtiges Vorgehen:
Rescaling auf alle Datensätze anwenden: Training, Test und Validierung.
Augmentierung nur auf das Trainingsset anwenden.

param_grid = {
    'learning_rate': [0.001,0.05,0.1],#0.05 beste ergebnisse als 0.01, 0.2, ab 0.05 nur noch 65%, 0.001 =88
    'epochs_list': [10,5],#da viele bilde, wenig epochen
    'layer1': [20,15,10],
    'layer2': [12,10,8],
    'layer3': [8,4],
    'Denselayer': [6]#Denselayer 3 ist besser als 5
}
Auswertung der besten Hyperparameter:
Die besten Ergebnisse, basierend auf Validierungsgenauigkeit und -verlust, ergeben sich aus den folgenden Kombinatione:
Lernrate:
0.001 bietet im Durchschnitt die besten Ergebnisse, da die Validierungsgenauigkeit in den meisten Fällen 99 % oder höher erreicht und der Verlust minimal ist. Höhere Lernraten (0.05, 0.1) führen oft zu Instabilitäten oder schlechteren Ergebnissen.
Layer-Struktur:
layer1 = 10, layer2 = 8, layer3 = 4: Diese Konfiguration liefert die höchste Validierungsgenauigkeit von 100 % (z. B. bei Lernrate = 0.001).
Auch Konfigurationen mit etwas größeren Layern, z. B. layer1 = 20, layer2 = 12, layer3 = 8, schneiden gut ab, aber kleinere Netzwerke scheinen effektiver zu sein, was auf eine Überanpassung bei größeren Architekturen hindeutet.
Epochenanzahl (epochs_list):
Mit 10 Epochs wurde die beste Leistung erzielt. Kürzere Trainingsläufe (z. B. 5 Epochs) erreichen ebenfalls gute Werte, aber tendenziell ist der Validierungsverlust etwas höher.
Denselayer:
Ein konstanter Wert von 6 für Denselayer scheint gut zu funktionieren, da dieser Wert in allen getesteten Konfigurationen konsistent war.

Feedback Gültas:
-Auf Rechner Trainieren -Fertig
-Scibo Bilder teilen -Fertig
-Attantion mechanism -Fertig
-Yolo als Vergleichs model
-Temperatur Schwankungen
-Masterarbeit grundstruktur schreiben
	- Warum Wichtig
	- Wann kommt das in frage
- gleichviele daten -fertig

verteilung = [0.2, 0.5, 0.3] ausprobieren neuer datensatz = Wenn Validation zu gering ist wird keine gute accouracy erreicht

neu laufen lassen mit augmentation in gridsearch-fertig
augmentated bilder anzeigen-fertig
verteilung = [0.2, 0.5, 0.3] #Test, Train, Validation = gut ? = Ja mann braucht min 30% validation

Todo:
(Yolo datensatz mit nur einem order
Datensatz
-Healty
-Dry)zuerst yolo_labeler ausprobieren

Yolo_labeler ausführen

Yolo env activirern 
Yolo ausführen

cnn konvolution metrik mit yolo vergleichen
complete2 mit 0.4 val

Treffen Feedback: 24.1.25
Yolo auf 100epochen
Latex = gut -> Menü -> Compiler -> PdfLatex
ResNet nachgucken
Latex in sciebo
Programm kommt in den anhnag der Masterarbeit
Marktanalyse = Ja
-> Praktische anwendung
-> Prototyp
-> Für jede Erdbeere?
-> Modell-Gewichte anpassen ?
Zotat Format irgendetwas mit Zahlen nehmen

Fragen Treffen 7.2.:
-Falsches gelernt? = Neue Pflanze = nein
-Anstatt ResNet EfficientNetB0? = nein ungenauer

Nächste woche freitag um 11:00
Am montag guckt er auf den code
Git repo

Git Repo
-Git Bash
-> cd C:/Users/maxbi/OneDrive/Dokumente/Masterstudiengang/Masterarbeit/Gültas
-> git clone https://maxbieger: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@github.com/maxbieger/Masterarbeit_Bieger.git

git Push:
-git Bash
->git status
->git add .
->git commit -m "Deine Commit-Nachricht"
->git push origin main



Feedback 14.2.:
-yolo Augmentierung einfügen, epochen=100
-schreiben anfangen
-anmeldung machen
-ResNet nur mit 1-2Sätzen im fazit erwähnen
-Nächstes Metting nächsten Freitag 11:00 nur mit Gültas
-Wieder link per Email schicken


Fragen:
-Folgende Zusatzfächer sollen auf meinem Zeugnis augegeben werden
-Thema genau schreiben
-ResNet = 94%
-Yolo =
-etwa 70 Seiten à 30 Zeilen (exklusive Bilder und
Tabellen
-21 Wochen
-Englische fachbegriffe JANein
-Antrag auf Zulassung zur Masterarbeit sind folgende weiteren
Unterlagen beizufügen:
a) eine Erklärung darüber, welche Module als Wahlpflichtmodule festgelegt werden
b) eine Erklärung darüber, ob die Masterarbeit abweichend von § 30 Absatz 4 RPO in englischer Sprache 
- Kolloquium: mindestens 30 und maximal 60 Minuten
- Wo Masterarbeit abgeben, in welcher Form