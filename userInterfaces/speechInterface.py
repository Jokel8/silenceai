import pyttsx3
import sys
from userInterfaces import consoleInterface


def say(text):
    try:
        if type(text) is not str:
            raise ValueError("Die Eingabe muss ein String sein.")

        engine = pyttsx3.init()
        engine.setProperty('rate', 200)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Fehler bei der Sprachausgabe: {e}")
        consoleInterface.print_error("Fehler bei der Sprachausgabe: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "Hallo, wie geht es dir? Ich hoffe, du hast einen großartigen Tag!"
        
    say(text)
    print("Sprachausgabe abgeschlossen.")
