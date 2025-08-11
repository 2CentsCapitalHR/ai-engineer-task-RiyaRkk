import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai

# ---------------- CONFIG ----------------
URL = "https://en.adgm.thomsonreuters.com/entiresection/1"
DB_DIR = "adgm_rules_db"
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
BATCH_SIZE = 50  # number of chunks per embedding API call

# ---------------- LOAD API KEY ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file!")

# Gemini API client
client = genai.Client(api_key=API_KEY)


# ---------------- CUSTOM GEMINI EMBEDDING FUNCTION ----------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB EmbeddingFunction using Google Gemini's embedding API.
    """

    def __init__(self, model_name: str, client, batch_size: int = 50):
        self.model_name = model_name
        self.client = client
        self.batch_size = batch_size  # for batching long lists

    def __call__(self, inputs: Documents) -> Embeddings:
        all_embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch
            )
            # response.embeddings is a list of embedding dicts
            for emb_obj in response.embeddings:
                all_embeddings.append(emb_obj.values)

        return all_embeddings


# ---------------- SCRAPE ADGM WEBSITE ----------------
def scrape_text(url: str) -> str:
    print(f"[INFO] Fetching: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove irrelevant tags
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.extract()

    text = soup.get_text(separator="\n")

    # Normalize whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# ---------------- CHUNK TEXT ----------------
def chunk_text(text: str):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if chunk:
            chunks.append(chunk)
    return chunks


# ---------------- STORE INTO CHROMADB ----------------
def store_embeddings(chunks):
    embedding_fn = GeminiEmbeddingFunction(
        model_name="models/text-embedding-004",  # Gemini latest embedding model
        client=client,
        batch_size=BATCH_SIZE
    )

    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_or_create_collection(
        name="adgm_rules",
        embedding_function=embedding_fn
    )

    # Store each chunk in collection
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"rule_{idx}"]
        )

    print(f"[INFO] ✅ Stored {len(chunks)} chunks in ChromaDB collection 'adgm_rules'.")


# ---------------- WRAPPER FOR STREAMLIT UI ----------------
def load_or_build_vector_db():
    """
    Wrapper function for the Streamlit UI.
    Loads existing 'adgm_rules' ChromaDB collection if it exists;
    otherwise scrapes and stores embeddings.
    """
    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    try:
        collection = chroma_client.get_collection("adgm_rules")
        count = collection.count()
        if count > 0:
            print(f"[INFO] Vector DB already exists with {count} records. Skipping ingestion.")
            return
    except Exception:
        # No existing collection — proceed to build it
        pass

    print("[INFO] Scraping ADGM rules...")
    full_text = scrape_text(URL)
    print("[INFO] Chunking text...")
    chunks = chunk_text(full_text)
    print(f"[INFO] Number of chunks created: {len(chunks)}")
    print("[INFO] Creating embeddings & storing in vector DB...")
    store_embeddings(chunks)
    print("[DONE] Ingestion pipeline completed successfully.")


# ---------------- MAIN SCRIPT ----------------
if __name__ == "__main__":
    print("[INFO] Scraping ADGM rules...")
    full_text = scrape_text(URL)

    print("[INFO] Chunking text...")
    chunks = chunk_text(full_text)
    print(f"[INFO] Number of chunks created: {len(chunks)}")

    print("[INFO] Creating embeddings & storing in vector DB...")
    store_embeddings(chunks)

    print("[DONE] Ingestion pipeline completed successfully.")
