from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.shortcuts import print_formatted_text

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_error(message):
    print_formatted_text(ANSI(f"{RED}{message}{RESET}\n"))

def print_instruction(message):
    print_formatted_text(ANSI(f"{YELLOW}{message}{RESET}\n"))

def print_status(message):
    print_formatted_text(ANSI(f"{BLUE}{message}{RESET}"))

def consoleLoop(state):
    session = PromptSession()
    
    while state.isRunning:
        with patch_stdout():
            user_input = session.prompt("\nGebe jederzeit einen Befehl ein: ")
        if user_input.lower() in ["exit", "quit"]:
            state.isRunning = False
        elif user_input.lower() == "help":
            print_instruction("Verfügbare Befehle:\n"
                         "  help - Zeigt diese Hilfe an\n"
                         "  exit, quit - Beendet das Programm\n"
                         "  toggle preprocessing - Aktiviert/Deaktiviert die Vorverarbeitung\n"
                         "  toggle postprocessing - Aktiviert/Deaktiviert die Nachverarbeitung\n"
                         "  toggle tts - Aktiviert/Deaktiviert die Sprachausgabe\n")
        elif user_input.lower() == "toggle preprocessing":
            state.usePreProcessing = not state.usePreProcessing
            status = "aktiviert" if state.usePreProcessing else "deaktiviert"
            print_instruction(f"Vorverarbeitung wurde {status}.")
        elif user_input.lower() == "toggle postprocessing":
            state.usePostProcessing = not state.usePostProcessing
            status = "aktiviert" if state.usePostProcessing else "deaktiviert"
            print_instruction(f"Nachverarbeitung wurde {status}.")
        elif user_input.lower() == "toggle tts":
            state.useTextToSpeech = not state.useTextToSpeech
            status = "aktiviert" if state.useTextToSpeech else "deaktiviert"
            print_instruction(f"Sprachausgabe wurde {status}.")
        else:
            print_error(f"Ungültige Eingabe: '{user_input}'. Gebe 'help' ein für eine Liste der Befehle.")