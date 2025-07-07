import fitz  # PyMuPDF

pdf_path = "s41746-025-01653-8.pdf"  # Update path if needed
output_txt = "s41746-025-01653-8.txt"

# Open the PDF
doc = fitz.open(pdf_path)

# Extract text from all pages
with open(output_txt, "w", encoding="utf-8") as out:
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        out.write(f"\n--- Page {page_num + 1} ---\n")
        out.write(text)

print(f"Text extracted to {output_txt}")