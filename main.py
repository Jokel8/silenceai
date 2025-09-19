import threading
import time
from textToSpeech import textToSpeech

isRunning = True
usePreProcessing = True
usePostProcessing = True
useTextToSpeech = True
gotGeasture = False

def textLoop():
    global isRunning
    
    while isRunning:
        user_input = input("Gebe jederzeit einen Command ein: ")
        if user_input.lower() in ["exit", "quit", "beenden"]:
            isRunning = False
        else:
            print(f"Fehler: '{user_input}'")

def captureLoop():
    global isRunning
    
    while isRunning:
        analyseThread = threading.Thread(target=analysis)
        analyseThread.start()
        time.sleep(3)
        
def analysis():
    global isRunning, usePostProcessing, useTextToSpeech, gotGeasture
    
    if gotGeasture: return
    gotGeasture = True
    print("Verarbeite Videosignal...")
    
    #KI Analyse
    text = "Test"
    #text, confidence = analyzeVideo()
    
    if not isRunning: return
    
    if usePostProcessing:
        print("Postprocessing wird angewendet...")
        #text = postProcessing(text)

    if not isRunning: return
    
    if useTextToSpeech:
        textToSpeech.say(text)
    
    gotGeasture = False
        
textThread = threading.Thread(target=textLoop)
textThread.start()

captureThread = threading.Thread(target=captureLoop)
captureThread.start()

# Warten, bis der Nebenthread beendet ist
textThread.join()
textThread.join()
print("SilenceAI wurde beendet")