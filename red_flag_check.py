import os
import re
import json
from pathlib import Path
import fitz
from docx import Document
from dotenv import load_dotenv
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from google import genai

# ---------------- CONFIG ----------------
DB_DIR = "adgm_rules_db"
TOP_K = 5
MODEL = "gemini-2.5-flash"
OUTPUT_JSON = "redflag_report.json"
OUTPUT_TXT = "redflag_agent3_input.txt"

# ---------------- API KEY ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env file!")

client = genai.Client(api_key=API_KEY)

# ---------------- EMBEDDING FUNCTION ----------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str, client):
        self.model_name = model_name
        self.client = client

    def __call__(self, inputs: Documents):
        all_embeddings = []
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=inputs
        )
        for emb_obj in response.embeddings:
            all_embeddings.append(emb_obj.values)
        return all_embeddings

# ---------------- TEXT EXTRACTION ----------------
def normalize(text: str):
    return re.sub(r"\s+", " ", text).strip()

def read_docx(path):
    doc = Document(path)
    return normalize("\n".join([p.text for p in doc.paragraphs if p.text.strip()]))

def read_pdf(path):
    parts = []
    with fitz.open(path) as pdf:
        for page in pdf:
            t = page.get_text()
            if t:
                parts.append(t.strip())
    return normalize("\n".join(parts))

def extract_text(path):
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return read_docx(path)
    elif ext == ".pdf":
        return read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ---------------- RETRIEVE RULES ----------------
def retrieve_rules(query_text):
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    emb_fn = GeminiEmbeddingFunction("models/text-embedding-004", client)
    collection = chroma_client.get_collection("adgm_rules", embedding_function=emb_fn)
    results = collection.query(query_texts=[query_text], n_results=TOP_K)

    # Flatten nested results
    docs_nested = results["documents"]  # e.g., [["chunk1", "chunk2", ...]]
    combined_docs = []
    for sublist in docs_nested:
        combined_docs.extend(sublist)
    return "\n\n".join(combined_docs)

# ---------------- CLEAN LLM OUTPUT ----------------
def clean_llm_output(text: str) -> str:
    """Remove markdown code fences (``````) from Gemini output"""
    cleaned = text.strip()
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()

# ---------------- DETECT RED FLAGS ----------------
def detect_red_flags(rules, doc_text):
    prompt = f"""
You are a compliance review assistant for ADGM registration.

ADGM Rules:
---
{rules}
---

Document:
---
{doc_text}
---

TASK:
Check ONLY for:
1. Invalid or missing clauses
2. Incorrect jurisdiction
3. Ambiguous or non-binding language
4. Missing signatory sections or improper formatting
5. Non-compliance with ADGM templates

OUTPUT STRICTLY IN JSON:
{{
"summary": "...",
"red_flags":[
{{"issue":"...", "law_reference":"...", "snippet":"..."}}
]
}}
"""
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    cleaned_text = clean_llm_output(resp.text)
    return json.loads(cleaned_text)

# ---------------- SAVE TSV FOR AGENT 3 ----------------
def save_agent3_friendly(red_flags, path):
    """Save snippet, issue, and law_reference as tab-separated lines for Agent 3."""
    with open(path, "w", encoding="utf-8") as f:
        for rf in red_flags:
            snippet = (rf.get("snippet") or "").replace("\t", " ")
            issue = (rf.get("issue") or "").replace("\t", " ")
            law_ref = (rf.get("law_reference") or "").replace("\t", " ")
            f.write(f"{snippet}\t{issue}\t{law_ref}\n")

# ---------------- WRAPPER FOR STREAMLIT UI ----------------
def check_red_flags(file_path):
    """
    Wrapper for Streamlit UI:
    - Extracts text from the uploaded file
    - Runs vector search on ADGM rules
    - Detects red flags with Gemini
    - Saves results as JSON & TSV
    - Returns (red_flags_list, json_path, tsv_path)
    """
    text = extract_text(file_path)
    rules = retrieve_rules(text)
    data = detect_red_flags(rules, text)

    json_path = OUTPUT_JSON
    tsv_path = OUTPUT_TXT
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, indent=2, ensure_ascii=False)

    save_agent3_friendly(data.get("red_flags", []), tsv_path)
    return data.get("red_flags", []), json_path, tsv_path

# ---------------- MAIN SCRIPT ----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} file.docx|file.pdf")
        sys.exit(1)

    doc_path = sys.argv[1]
    if not Path(doc_path).exists():
        print(f"[ERROR] File not found: {doc_path}")
        sys.exit(1)

    print("[INFO] Extracting document text...")
    text = extract_text(doc_path)

    print("[INFO] Retrieving relevant ADGM rules...")
    rules = retrieve_rules(text)

    print("[INFO] Running LLM red flag detection...")
    data = detect_red_flags(rules, text)

    print(f"[INFO] Saving outputs: {OUTPUT_JSON} and {OUTPUT_TXT}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(data, jf, indent=2, ensure_ascii=False)
    save_agent3_friendly(data.get("red_flags", []), OUTPUT_TXT)

    print("[DONE] Agent 2 finished.")
    print(f" JSON report: {OUTPUT_JSON}")
    print(f" Agent3 input: {OUTPUT_TXT}")
