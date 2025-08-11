import os
import re
import fitz  # PyMuPDF for PDFs
from docx import Document
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from mapping_table import mapping_table  # ✅ to map doc_type → checklist file or URL

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment variables.")

# Initialize Gemini Client
client = genai.Client(api_key=API_KEY)

# ---------------------------
# Text Extraction Utilities
# ---------------------------
def normalize(text: str) -> str:
    """Normalize whitespace and newlines for better LLM input."""
    return re.sub(r"\s+", " ", text).strip()

def read_docx(file_path):
    """Extract cleaned text from DOCX files."""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return normalize(text)

def read_pdf(file_path):
    """Extract cleaned text from PDF files using PyMuPDF."""
    text = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            page_text = page.get_text().strip()
            if page_text:
                text.append(page_text)
    return normalize("\n".join(text))

def extract_text(file_path):
    """Extract normalized text from supported file types."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        return read_docx(file_path)
    elif suffix == ".pdf":
        return read_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

# ---------------------------
# Gemini LLM Comparison Function
# ---------------------------
def compare_with_gemini(checklist_text, uploaded_text):
    """
    Use the Gemini chat to compare checklist vs uploaded document,
    and return missing documents info.
    """
    prompt = f"""
    You are a compliance assistant. Compare the REQUIRED DOCUMENT CHECKLIST
    to the UPLOADED DOCUMENT content.

    TASK:
    1. Identify all required documents/items mentioned in the checklist.
    2. Check if each required item is present in the uploaded document.
    3. List each missing item clearly under "MISSING DOCUMENTS".

    Return your answer strictly in this format:
    SUMMARY:
    MISSING DOCUMENTS:

    - ...

    --- CHECKLIST CONTENT ---
    {checklist_text}

    --- UPLOADED DOCUMENT CONTENT ---
    {uploaded_text}
    """
    chat = client.chats.create(model="gemini-2.5-flash")  # Adjust if needed
    response = chat.send_message(message=prompt)
    return response.text

# ---------------------------
# New Wrapper for Streamlit UI
# ---------------------------
def find_missing_documents(uploaded_file_path, doc_type):
    """
    Wrapper for Streamlit integration.
    Takes the uploaded document path and Step1's classification doc_type,
    finds the corresponding checklist file or URL, compares and returns missing items.
    """
    # 1. Get checklist file or URL from mapping_table
    checklist_source = mapping_table.get(doc_type)
    if not checklist_source:
        return []  # No checklist source found

    # If it's a URL, you might need to download it or handle differently.
    # Here, we assume it's a local file path.
    checklist_path = Path(checklist_source)
    if not checklist_path.exists():
        print(f"[WARN] Checklist path for '{doc_type}' not found locally: {checklist_path}")
        return []

    # 2. Extract texts
    checklist_text = extract_text(checklist_path)
    uploaded_text = extract_text(uploaded_file_path)

    # 3. Compare with Gemini
    comparison_result = compare_with_gemini(checklist_text, uploaded_text)

    # 4. Parse Gemini's output into a list of missing items
    missing_items = []
    capture = False
    for line in comparison_result.splitlines():
        if line.strip().upper().startswith("MISSING DOCUMENTS"):
            capture = True
            continue
        if capture:
            if line.strip().startswith("-"):
                missing_items.append(line.strip("- ").strip())

    return missing_items

# ---------------------------
# Standalone execution for CLI
# ---------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} checklist_file uploaded_file")
        exit(1)

    script_dir = Path(__file__).parent
    checklist_file = script_dir / sys.argv[1]
    uploaded_file = script_dir / sys.argv[2]

    if not checklist_file.exists():
        print(f"[ERROR] Checklist file not found: {checklist_file}")
        exit(1)
    if not uploaded_file.exists():
        print(f"[ERROR] Uploaded file not found: {uploaded_file}")
        exit(1)

    print("[INFO] Extracting checklist text...")
    checklist_text = extract_text(checklist_file)

    print("[INFO] Extracting uploaded document text...")
    uploaded_text = extract_text(uploaded_file)

    print("[INFO] Sending data to Gemini LLM for comparison...")
    result = compare_with_gemini(checklist_text, uploaded_text)

    print("\n===== AI COMPARISON RESULT =====")
    print(result)
