# silenceai
**Translator for DGS**
Gebärdensprache ist eine komplexe und ausdrucksstarke Kommunikationsform, die für Millionen Menschen weltweit essenziell ist. Dennoch gibt es in vielen Situationen Kommunikationsbarrieren zwischen Menschen, die Gebärdensprache verwenden, und jenen, die diese nicht beherrschen. Unser Projekt SilentAI entwickelt ein KI-gestütztes Übersetzungsprogramm, das Gebärdensprache in Echtzeit in gesprochene und geschriebene Sprache übersetzt.

**Unsere Ziele:**
1.	Automatische Erkennung direkt über eine Kamera
2.	Echtzeitübersetzung mit Sprachausgabe
3.	Bei genug Zeit: Benutzerfrendliche Schnittstelle als Webapp oder Android App

**Unsere Vorgehensweise:**
1.  Pre-Processing (Hintergrund entfernen, Helligkeit und Kontrast normalisieren) @Jan
2.  Keypoint-Erkennung (Skelett der Hände berechnen) mit MediaPipe Holistic @Jonas
3.  Gebärdenerkennung durch ein eigenes Modell mit den RWTH Datensätzen @Jonas
4.  Post-Processing (Grammatikalische Korrektur und ggf. einfügen von Präpositionen) mit Qwen2 0.5B Parameter @Kende
