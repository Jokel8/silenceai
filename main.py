import threading
import time
from userInterfaces import speechInterface, consoleInterface, graficInterface

class State():
    def __init__(self):
        self.isRunning = True
        self.usePreProcessing = True
        self.usePostProcessing = True
        self.useTextToSpeech = True
        self.gotGeasture = False
        self.guesses = [["", 0.0], ["", 0.0], ["", 0.0]]
        self.output = ""

def captureLoop(state):    
    consoleInterface.print_status("Starte Verarbeitung...")
    while state.isRunning:
        analyseThread = threading.Thread(target=analysis(state))
        analyseThread.start()
        time.sleep(3)
        
def analysis(state):    
    if state.gotGeasture: return
    state.gotGeasture = True
    consoleInterface.print_status("Verarbeite Videosignal...")
    time.sleep(1)
    
    #KI Analyse
    text = "Test"
    #text, confidence = analyzeVideo()
    
    if not state.isRunning: return
    
    if state.usePostProcessing:
        consoleInterface.print_status("Postprocessing wird angewendet...")
        #text = postProcessing(text)

    if not state.isRunning: return
    
    if state.useTextToSpeech:
        speechInterface.say(text)
    
    state.gotGeasture = False

state = State()

consoleThread = threading.Thread(target=consoleInterface.consoleLoop, args=(state,))
consoleThread.start()

app = graficInterface.MyApp(state)
app.run()

# captureThread = threading.Thread(target=captureLoop, args=(state,))
# captureThread.start()

# Warten, bis der Nebenthread beendet ist
consoleThread.join()
consoleInterface.print_instruction("SilenceAI wurde beendet")
exit(0)