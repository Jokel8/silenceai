Was ist das Ziel des Projekts? (max. 400 Zeichen):
Silence AI soll Gebärdensprache automatisch in Echtzeit in Text und Sprache übersetzen. Unser Ziel ist es, Kommunikationsbarrieren abzubauen und stummen Personen zu ermöglichen, direkt und ohne Dolmetscher mit anderen zu sprechen – im Alltag, bei Arztgesprächen, in Schulen und bei Behörden. 

Wer kann eure Ergebnisse verwenden? (max. 260 Zeichen):
Gehörlose Menschen, ihre Angehörigen, Fachkräfte in Medizin, Bildung und Pflege sowie alle, die spontan mit Gebärdensprache kommunizieren möchten. Silence AI soll direkte Verständigung ermöglichen, ohne auf Dolmetscher angewiesen zu sein. 

Beschreibung eures Datensatzes (max. 400 Zeichen):
Wir nutzen 28 GB öffentlich verfügbare Gebärdensprachdaten der RWTH Aachen (u. a. Tagesschau und Heute). Der Datensatz enthält über 1200 Gebärden und Buchstaben. Vor dem Training haben wir ihn in ca. 2 GB Keypoint-Koordinaten umgewandelt, die über Mediapipe extrahiert wurden. Das Projekt der RWTH Aachen beinhaltet zudem noch weitere Validierungsdaten und wir haben auch zu Beginn ein paar Bilder von uns selbst zu Testzwecken aufgenommen. 

Aufbereitung der Daten und sonstige Vorbereitungen (max. 400 Zeichen):
Mittels einem Verzeichnis aus Beschreibungen, welches bei den Datensätzen enthalten war, werden einzelne Bilder aus den Video-Aufzeichnungen in entsprechende Ordner sortiert. In der Googles Mediapipe-Holistic-Libary nutzen wir ein Hand-Tracking-Modell und speichern danach alle Datensätze in Form von Koordinaten in passend benannten CSV-Dateien. Aus diesen absoluten Koordinaten werden danach relative Koordinaten zu dem Handballen berechnet und anschließend normalisiert, sodass keinen Zahlen betragsmäßig größer als eins entstehen. 

Beschreibung eurer Methoden (max. 400 Zeichen):
Das System verwendet ein vierschichtiges neuronales Netz für die Handgestenerkennung, das mittels TensorFlow implementiert ist. Der Datenprozess umfasst die parallele Verarbeitung von CSV-Dateien mit normalisierten Handkoordinaten. Das Modelltraining erfolgt mit einem Sequential Neural Network bestehend aus fünf Dense Layers mit abnehmender Neuronenzahl (512→64) und Dropout-Schichten zur Overfitting-Prävention. Die Inferenz erfolgt in Echtzeit über die Webcam, wobei MediaPipe die Handlandmarken extrahiert und das trainierte Modell diese klassifiziert. Zur Optimierung kommen Early Stopping, Model Checkpointing und TensorBoard-Visualisierung zum Einsatz. 

Wie habt ihr euer Projekt ausgewertet? Welche Genauigkeit habt ihr auf den Trainingsdaten und auf den Testdaten erreicht? (max. 1000 Zeichen):
Wir haben unsere Pipeline mit Fokus auf die Trainingsdaten evaluiert. Nach Preprocessing und Keypoint-Extraktion mit Mediapipe wurden die normalisierten Koordinaten in unser neuronales Netz eingespeist. Dabei erreichten wir auf den Trainingsdaten eine Accuracy von 0,2733. Auch wenn dieser Wert noch niedrig ist, zeigt er, dass das Modell bereits erste Muster erkennen kann und die gewählte Architektur grundsätzlich funktioniert. Für die Bewertung nutzten wir Accuracy-Metriken und Live-Tests über die Webcam. Die Ergebnisse liefern uns eine gute Grundlage, um durch mehr Daten und Feintuning die Genauigkeit in Zukunft deutlich zu verbessern. 

Was habt ihr entwickelt? (max. 260 Zeichen):
Ein KI-System und eine Grafische Oberfläche, bestehend aus 5 KIs:  
-   Extraktion der Person und entfernen des Hintergrunds 
-   Erkennung der Hände und Extraktion von Hand-Keypoints durch die Mediapipe-Libary 
-   Normalisierung der Koordinaten und ein neuronales Netz zur Gebärdenerkennung 
-   Postprocessing mit Gemma-2.5-7B für korrekte Sätze 
-   Und schließlich eine Text-to-Speech-Ausgabe. 

Was benötigt man, um euer Ergebnis nutzen zu können? (max. 260 Zeichen): 
Einen PC oder Laptop mit Webcam, Python, Mediapipe, Tensorflow und einigen kleineren Libarys (definiert in der requirements.md) sind die minimalen Voraussetzungen, um Testing.py zum Laufen zu kriegen. Zudem kann Tansformers, KIVY und PyTTSx3, wenn man noch mehr Features haben möchte und die Pipeline.py oder sogar die Main.py zum laufen bringen möchte. Der Code steht auf GitHub zur Verfügung unter https://github.com/Jokel8/silenceai.git . 

Auf welche Probleme seid ihr gestoßen? (max. 400 Zeichen):
Herausfordernd waren vor allem die Trainingsdaten, da diese in sehr geringer Auflösung und mit hoher Bewegungsunschärfe waren und wir dabei vermutlich 2/3 verloren haben, da oft nur eine Hand oder keine Hand zu erkennen waren. Zudem hatten wir bei zu hellen Umgebungen Schwierigkeiten die Hände zu erkennen. 

Was ist das größte Potential eures Projekts? (max. 260 Zeichen):
Mit Silence AI könnte Gebärdensprache überall sofort verständlich werden – von spontanen Alltagsgesprächen bis hin zu Behördengängen oder Arztbesuchen. Das eröffnet neue Chancen für Teilhabe und reduziert Abhängigkeit von Dolmetschern. Zudem soll es in Zukunft eine Android-App werden, in die jeder Entwickler sein eigenes Modell laden kann, um so eine Plattform für die Zukunft zu schaffen. 

Was ist die größte Schwachstelle eures Projekts? (max. 260 Zeichen):
Die größte Schwachstelle liegt in der noch geringen Genauigkeit der Gebärdenerkennung, da wir es noch nicht geschafft haben unser Model auch auf die Bewegungsabläufe zu trainieren, und da oft eine der Hände fehlt. Außerdem ist das System bislang nur als PC-Prototyp verfügbar und nicht mobil einsetzbar, da sich die Portierung als sehr schwer auf. 

Wie würdet ihr euer Projekt vorantreiben, wenn ihr unendlich viele Ressourcen und Zeit hättet? (max. 400 Zeichen):
Unendlich Ressourcen würden uns erlauben, die Erkennung auf fast alle Gebärden auszuweiten, eine hochpräzise KI zu entwickeln und Silence AI weltweit als App verfügbar zu machen. Ein Werkzeug für echte Inklusion ohne Sprachbarrieren.

Angabe von genutzten Quellen:
    Trainingsdaten:
    O. Koller, J. Forster, and H. Ney. Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. Computer Vision and Image Understanding, volume 141, pages 108-125, December 2015.
    Koller, Zargaran, Ney. "Re-Sign: Re-Aligned End-to-End Sequence Modeling with Deep Recurrent CNN-HMMs" in CVPR 2017, Honululu, Hawaii, USA.

    Tutorials:
    https://youtu.be/wa2ARoUUdU8?si=ze8F6maBpyoUMv4A
    https://youtu.be/a99p_fAr6e4?si=dXsEDock9pYthBFt

    Eingebettete KIs:
    cvzone.SelfiSegmentationModule, Mediapipe, PyTTSx3, Qwen7B, Gemma3 270M

    Genutzte Generative KIs
    VSCode Code-Completion, VSCode Copilot (ChatGPT4.1, ChatGPT5 und Claude Sonnet 3.7)
    Zudem ElevenLabs und DALL-E