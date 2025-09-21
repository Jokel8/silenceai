from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Pfad zu deinem .safetensors Modell (lokal entpackt)
model_path = "SilenceAI/postprocessing/gemma-3-270m/"

# Tokenizer und Modell laden
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto"
)

# Prompt
prompt = "Erkläre mir kurz, warum der Himmel blau ist."

# Eingabe vorbereiten
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Ausgabe generieren
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

# Ausgabe decodieren
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Output: ", output_text)
