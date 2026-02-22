import sys
import subprocess
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import fitz
except ImportError:
    install('pymupdf')
    import fitz

pdf_path = r"C:\Users\abhin\Downloads\Gen_Ai_Robustness.pdf"
out_txt = r"C:\Users\abhin\OneDrive\Documents\GitHub\CVIP\pdf_content.txt"

try:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print("PDF extracted successfully")
except Exception as e:
    print("Failed to extract:", e)
