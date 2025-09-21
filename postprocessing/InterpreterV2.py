#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InterpreterV2.py
----------------
LLM-basierter Gloss→Deutsch-Interpreter (Hugging Face Transformers).

- Eingabe: Gloss in ALL CAPS (z. B. "MORGEN ICH GEHEN ARZT")
- Ausgabe: Ein einziger korrekt formulierter deutscher Satz.
- Default-Modell: Qwen/Qwen2.5-1.5B-Instruct (frei zugänglich).
- Alternativ: google/gemma-3-270m-it (gated, Token/Login nötig).

CLI:
  python InterpreterV2.py --text "MORGEN ICH GEHEN ARZT"
  python InterpreterV2.py --stdin
"""

from __future__ import annotations
import argparse
import os
import re
from typing import List, Dict, Optional

# ----------------------------
# Prompt-Template
# ----------------------------

PROMPT_SYSTEM = (
    "Du bist ein Assistent. "
    "Deine Aufgabe: Verwandle DGS-ähnliche Gloss-Notation (ALL CAPS) "
    "in einen grammatisch korrekten, natürlichen deutschen Satz. "
    "WICHTIG: Gib NUR den Satz aus, ohne zusätzliche Wörter oder Erklärungen."
)

PROMPT_USER_TEMPLATE = (
    "Eingabe (Gloss in GROSSBUCHSTABEN):\n{gloss}\n\n"
    "Ausgabe (ein einzelner korrekter deutscher Satz):"
)

FEWSHOTS: List[Dict[str, str]] = [
    {"role": "user", "content": PROMPT_USER_TEMPLATE.format(gloss="MORGEN ICH GEHEN ARZT")},
    {"role": "assistant", "content": "Morgen gehe ich zum Arzt."},
    {"role": "user", "content": PROMPT_USER_TEMPLATE.format(gloss="GESTERN WIR FAHREN KINO")},
    {"role": "assistant", "content": "Gestern sind wir ins Kino gefahren."},
    {"role": "user", "content": PROMPT_USER_TEMPLATE.format(gloss="JETZT ICH MICH HINLEGEN")},
    {"role": "assistant", "content": "Jetzt lege ich mich hin."},
    {"role": "user", "content": PROMPT_USER_TEMPLATE.format(gloss="DU TRINKEN WASSER")},
    {"role": "assistant", "content": "Du trinkst Wasser."},
    {"role": "user", "content": PROMPT_USER_TEMPLATE.format(gloss="ICH WOHNEN BERLIN")},
    {"role": "assistant", "content": "Ich wohne in Berlin."},
]

# ----------------------------
# HF Pipeline
# ----------------------------

_HF_PIPE = None

def _init_hf_pipeline(model_name: str, hf_token: Optional[str] = None):
    global _HF_PIPE
    if _HF_PIPE is not None:
        return _HF_PIPE

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN", None)

    tok = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    _HF_PIPE = pipeline("text-generation", model=model, tokenizer=tok)
    return _HF_PIPE

def _build_prompt(gloss: str) -> str:
    lines: List[str] = []
    lines.append(f"[System]\n{PROMPT_SYSTEM}\n")
    for shot in FEWSHOTS:
        role = shot["role"]
        content = shot["content"]
        lines.append(f"[{role.capitalize()}]\n{content}\n")
    lines.append(f"[User]\n{PROMPT_USER_TEMPLATE.format(gloss=gloss)}\n")
    lines.append("[Assistant]\n")
    return "\n".join(lines)

def _strip_dialog_markers(text: str) -> str:
    for marker in ("Human:", "Assistant:", "User:", "System:"):
        if marker in text:
            text = text.split(marker)[0].strip()
    return text

def _first_sentence(text: str) -> str:
    """
    Schneidet nach dem ersten vollständigen Satz ab.
    Nimmt das erste Vorkommen von '.', '!' oder '?' nach einem Wortanfang.
    """
    text = text.strip()
    # Versuch 1: Regex – alles bis inkl. erstem Satzende
    m = re.search(r'^[\s"„‚»«]*([A-Za-zÄÖÜäöüß0-9][^.!?]*[.!?])', text)
    if m:
        return m.group(1).strip().strip('"„“‚‘»«')
    # Versuch 2: Fallback – bis erstes Satzzeichen
    for ch in ('.', '!', '?'):
        if ch in text:
            return text.split(ch)[0].strip() + ch
    # Versuch 3: Fallback – hart kürzen
    return (text[:140].rstrip() + '.') if text else ""

# ----------------------------
# API
# ----------------------------

def interpret_gloss(
    gloss: str,
    hf_model: str = "Qwen/Qwen2.5-7-B-Instruct",
    max_tokens: int = 64,
    hf_token: Optional[str] = None,
) -> str:
    """
    Gloss (ALL CAPS) -> deutscher Satz (Hugging Face Transformers).
    Es werden absichtlich KEINE Sampling-Parameter übergeben,
    damit keine 'invalid generation flags'-Warnungen erscheinen.
    """
    pipe = _init_hf_pipeline(hf_model, hf_token=hf_token)

    gloss_clean = " ".join(gloss.strip().split())
    prompt = _build_prompt(gloss_clean)

    # Nur robuste, universell akzeptierte Parameter übergeben
    out = pipe(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    text = out[0]["generated_text"]

    # 1) Chat-Vorspann abschneiden
    if "[Assistant]" in text:
        text = text.split("[Assistant]")[-1].strip()

    # 2) Dialogmarker entfernen (Human:, Assistant: ...)
    text = _strip_dialog_markers(text)

    # 3) Nur den ersten vollständigen Satz behalten
    text = _first_sentence(text)

    return text

# ----------------------------
# CLI
# ----------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Interpreter V2 (LLM, HF) – Gloss → Deutsch")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", "-t", type=str, help="Gloss in ALL CAPS")
    src.add_argument("--stdin", action="store_true", help="Interaktiv: Gloss-Zeilen eingeben")

    p.add_argument("--hf-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF Modell-ID (z. B. Qwen/Qwen2.5-1.5B-Instruct; gated: google/gemma-3-270m-it)")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HF Access Token (für gated Modelle nötig)")
    p.add_argument("--max-tokens", type=int, default=64)
    return p.parse_args()

def main():
    args = _parse_args()

    if args.text:
        print(interpret_gloss(
            args.text,
            hf_model=args.hf_model,
            max_tokens=args.max_tokens,
            hf_token=args.hf_token,
        ))
    else:
        print("Gloss eingeben (ALL CAPS). Leere Zeile beendet.")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            print(interpret_gloss(
                line,
                hf_model=args.hf_model,
                max_tokens=args.max_tokens,
                hf_token=args.hf_token,
            ))

if __name__ == "__main__":
    main()
