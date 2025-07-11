import os
import json
import time
from dotenv import load_dotenv
import requests

# Load API key from .env or environment variable
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment or .env file.")

INPUT_JSON = "textbook_chunks.json"
OUTPUT_JSON = "qa_cot_data.json"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY

PROMPT_TEMPLATE = (
    "Given the following medical textbook excerpt, generate ONE USMLE-style multiple-choice question, 4 answer options (A, B, C, D), the correct answer, and a detailed step-by-step explanation (chain-of-thought) for the answer.\n"
    "Textbook Excerpt:\n{chunk}\n\n"
    "Output format:\n"
    "Question: ...\n"
    "Options:\nA) ...\nB) ...\nC) ...\nD) ...\n"
    "Answer: ...\n"
    "Explanation: ...\n"
)

def call_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 512
        }
    }
    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print("Error parsing Gemini response:", e)
            return None
    else:
        print(f"Gemini API error {response.status_code}: {response.text}")
        return None

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    output = []
    # Only process the first 150 chunks for proof of concept
    chunks_to_process = chunks[:150]
    for i, chunk in enumerate(chunks_to_process):
        chunk_text = chunk["text"]
        prompt = PROMPT_TEMPLATE.format(chunk=chunk_text)
        print(f"Processing chunk {i+1}/{len(chunks_to_process)}...")
        qa_cot = call_gemini_api(prompt)
        if qa_cot:
            output.append({
                "chapter_title": chunk["chapter_title"],
                "chunk_index": chunk["chunk_index"],
                "qa_cot": qa_cot
            })
        else:
            output.append({
                "chapter_title": chunk["chapter_title"],
                "chunk_index": chunk["chunk_index"],
                "qa_cot": None
            })
        time.sleep(1)  # avoid hitting rate limits
        # Optional: Save progress every 50 chunks
        if (i+1) % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
    # Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Done. Saved {len(output)} QA-CoT entries to {OUTPUT_JSON}.")

if __name__ == "__main__":
    main() 