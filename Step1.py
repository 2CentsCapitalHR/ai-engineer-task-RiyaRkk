import os
import json
import difflib
import requests
from urllib.parse import urljoin, urlparse
from collections import deque
from dotenv import load_dotenv
from google import genai
from docx import Document
from bs4 import BeautifulSoup
from mapping_table import mapping_table  # ✅ your mappings

# === Load API key from .env ===
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

# ================== Utils ==================
def extract_text_from_docx(file_path):
    """Extract text content from a DOCX file."""
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def find_closest_mapping_key(ai_classified_type, mapping_keys):
    """Find the closest mapping table key to the AI output using fuzzy match."""
    matches = difflib.get_close_matches(ai_classified_type, mapping_keys, n=1, cutoff=0.5)
    if matches:
        return matches[0]
    return None


# ================== AI helpers ==================
def identify_document_type_ai(document_text, doc_types):
    """Ask Gemini to classify the uploaded document into one of the mapping types."""
    prompt = f"""
    You are a document classifier. Choose ONE exact document type from the provided list.

    Document text:
    {document_text}

    Possible document types:
    {', '.join(doc_types)}

    Return ONLY the exact matching type from the list.
    """
    chat = client.chats.create(model="gemini-1.5-flash")
    response = chat.send_message(prompt)
    return response.text.strip()


def filter_checklist_docs(candidates, document_text):
    """AI filter to select checklists, required document lists, guidelines, or procedural manuals."""
    filtered = []
    for doc in candidates:
        prompt = f"""
        You are selecting official documents useful for verifying or preparing the uploaded document.

        These may include:
        - Checklists
        - Lists of required documents
        - Guidelines
        - Instructions
        - Procedural manuals

        Uploaded document:
        {document_text}

        Candidate document:
        Title: {doc['title']}
        URL: {doc['url']}

        If you think this could be even partially helpful for verification, include it.

        Respond in JSON:
        {{
        "decision": "include" or "exclude",
        "summary": "short reason if included"
        }}
        """
        try:
            chat = client.chats.create(model="gemini-1.5-flash")
            resp = chat.send_message(prompt)
            data = json.loads(resp.text.strip())
            if data.get("decision") == "include":
                doc["summary"] = data.get("summary", "")
                filtered.append(doc)
        except Exception:
            pass
    return filtered


# ================== Scraping ==================
def scrape_documents_single_page(url):
    """Scrape only one page for direct document links."""
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        doc_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(href.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
                full_link = href if href.startswith('http') else urljoin(url, href)
                title = link.get_text(strip=True) or os.path.basename(full_link)
                doc_links.append({"title": title, "url": full_link})
        return doc_links
    except Exception as e:
        print(f"Scraping error: {e}")
        return []


def scrape_documents_recursive(start_url, max_depth=2):
    """Recursive crawler to collect files up to max_depth internal links deep."""
    visited = set()
    queue = deque([(start_url, 0)])
    doc_links = []
    domain = urlparse(start_url).netloc

    while queue:
        current_url, depth = queue.popleft()
        if current_url in visited or depth > max_depth:
            continue
        visited.add(current_url)
        try:
            resp = requests.get(current_url, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_link = href if href.startswith('http') else urljoin(current_url, href)

                # Collect documents
                if any(full_link.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
                    title = link.get_text(strip=True) or os.path.basename(full_link)
                    doc_links.append({"title": title, "url": full_link})

                # Queue internal HTML pages
                elif domain in urlparse(full_link).netloc:
                    queue.append((full_link, depth + 1))
        except Exception as e:
            print(f"Scraping error at {current_url}: {e}")

    # Deduplicate by URL
    unique_links = {link['url']: link for link in doc_links}
    return list(unique_links.values())


# ================== Main pipeline ==================
def main(docx_path, deep_scrape=True, crawl_depth=2):
    # Extract text from uploaded doc
    document_text = extract_text_from_docx(docx_path)

    # Classification
    doc_type_raw = identify_document_type_ai(document_text, list(mapping_table.keys()))
    doc_type = find_closest_mapping_key(doc_type_raw, list(mapping_table.keys())) or doc_type_raw

    # Map to URL
    official_url = mapping_table.get(doc_type)
    if not official_url:
        return {
            "identified_document_type": doc_type,
            "official_url": None,
            "checklist_documents": []
        }

    # Direct file handling
    if any(official_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
        return {
            "identified_document_type": doc_type,
            "official_url": official_url,
            "checklist_documents": [{
                "title": os.path.basename(official_url.split('?')[0]),
                "url": official_url,
                "summary": "Direct official document (no scraping needed)."
            }]
        }

    # Scraping
    if deep_scrape:
        candidates = scrape_documents_recursive(official_url, max_depth=crawl_depth)
    else:
        candidates = scrape_documents_single_page(official_url)

    # AI checklist filtering
    checklist_docs = filter_checklist_docs(candidates, document_text)
    return {
        "identified_document_type": doc_type,
        "official_url": official_url,
        "checklist_documents": checklist_docs
    }


# ================== Wrapper for UI ==================
def classify_document(file_path):
    """
    Wrapper for Streamlit UI.
    Runs only the classification step from Step1.py and returns just the document type.
    Ignores any URL or scraping results.
    """
    document_text = extract_text_from_docx(file_path)
    doc_type_raw = identify_document_type_ai(document_text, list(mapping_table.keys()))
    doc_type = find_closest_mapping_key(doc_type_raw, list(mapping_table.keys())) or doc_type_raw
    return doc_type


# ================== Run example ==================
if __name__ == "__main__":
    input_docx_path = "input_document.docx"
    output = main(input_docx_path, deep_scrape=True, crawl_depth=2)
    print(json.dumps(output, indent=4))
