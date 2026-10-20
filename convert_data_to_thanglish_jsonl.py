# -*- coding: utf-8 -*-
"""
Generate training dataset from data/*.txt.
Converts Tamil text to Thanglish and formats each record into:
{
  "prompt": [
    {
      "role": "system",
      "content": "பயனர் வழங்கும் தங்கிலீஷ் உரையை இயல்பான தமிழ் எழுத்தாக மாற்றவும். விளக்கம் அல்லது கூடுதல் தகவல் இல்லாமல் மாற்றப்பட்ட தமிழ் உரையை மட்டும் வழங்கவும்."
    },
    {
      "role": "user",
      "content": "naan veetuku poren"
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "நான் வீட்டுக்குப் போகிறேன்."
    }
  ]
}

Generates:
1. Individual .jsonl files in 'output/thanglish_pair_{index:07d}.jsonl'
2. Combined 'train.jsonl' in workspace root
"""

import os
import glob
import json
import re
from pathlib import Path
from thanglish_transliterator import to_thanglish

SYSTEM_PROMPT = "பயனர் வழங்கும் தங்கிலீஷ் உரையை இயல்பான தமிழ் எழுத்தாக மாற்றவும். விளக்கம் அல்லது கூடுதல் தகவல் இல்லாமல் மாற்றப்பட்ட தமிழ் உரையை மட்டும் வழங்கவும்."

WORKSPACE_DIR = Path(__file__).resolve().parent   # repo root; data/ is read from and output/ + train.jsonl written to here
DATA_DIR = WORKSPACE_DIR / "data"
OUTPUT_DIR = WORKSPACE_DIR / "output"
TARGET_COUNT = 903

def clean_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def collect_sentences():
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    sentences = []
    seen = set()

    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue
        
        # Split into sentence-like chunks
        parts = re.split(r'[\n\r]+', content)
        for part in parts:
            part = clean_sentence(part)
            if not part:
                continue
            # Filter reasonable sentence lengths (at least 8 chars, at most 300)
            if 8 <= len(part) <= 300:
                # Ensure it has Tamil characters
                if any('\u0b80' <= c <= '\u0bff' for c in part):
                    if part not in seen:
                        seen.add(part)
                        sentences.append(part)
                        if len(sentences) >= TARGET_COUNT:
                            return sentences
    return sentences

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean previous output if any
    for existing in OUTPUT_DIR.glob("thanglish_pair_*.*"):
        existing.unlink(missing_ok=True)
        
    print(f"Reading from {DATA_DIR}...")
    sentences = collect_sentences()
    print(f"Extracted {len(sentences)} unique Tamil sentences.")
    
    combined_records = []
    
    for idx, tamil_text in enumerate(sentences, start=1):
        thanglish_text = to_thanglish(tamil_text)
        
        record = {
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": thanglish_text
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": tamil_text
                }
            ]
        }
        
        # Save individual file as .jsonl (and .txt compatibility if needed)
        jsonl_path = OUTPUT_DIR / f"thanglish_pair_{idx:07d}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as out_f:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
        combined_records.append(record)

    # Save combined train.jsonl
    combined_path = WORKSPACE_DIR / "train.jsonl"
    with open(combined_path, "w", encoding="utf-8") as out_comb:
        for r in combined_records:
            out_comb.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {len(sentences)} files in {OUTPUT_DIR}")
    print(f"Combined dataset saved to: {combined_path}")

if __name__ == "__main__":
    main()
