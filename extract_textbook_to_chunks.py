import fitz  # PyMuPDF
import re
import json

PDF_PATH = "2022, CURRENT Medical Diagnosis and Treatment- Original.pdf"
OUTPUT_JSON = "textbook_chunks.json"
CHUNK_SIZE = 4000  # characters
CHUNK_OVERLAP = 200  # characters

# 1. Extract all text from the PDF
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# 2. Split text into chapters using regex (customize pattern as needed)
def split_into_chapters(text):
    # Example: chapters start with 'Chapter' or 'CHAPTER' followed by number/title
    pattern = re.compile(r"(Chapter\s+\d+[:\s].*|CHAPTER\s+\d+[:\s].*)", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    chapters = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        chapter_title = match.group().strip()
        chapter_text = text[start:end].strip()
        chapters.append({"chapter_title": chapter_title, "text": chapter_text})
    return chapters

# 3. Chunk each chapter

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

# 4. Main processing

def main():
    print("Extracting text from PDF...")
    full_text = extract_pdf_text(PDF_PATH)
    print("Splitting into chapters...")
    chapters = split_into_chapters(full_text)
    print(f"Found {len(chapters)} chapters.")
    output = []
    for chapter in chapters:
        chapter_title = chapter["chapter_title"]
        chapter_text = chapter["text"]
        chunks = chunk_text(chapter_text)
        for idx, chunk in enumerate(chunks):
            output.append({
                "chapter_title": chapter_title,
                "chunk_index": idx,
                "text": chunk
            })
    print(f"Writing {len(output)} chunks to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == "__main__":
    main() 