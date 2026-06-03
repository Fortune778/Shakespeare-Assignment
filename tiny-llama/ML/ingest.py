print("startt")
import os
print("loaded")
import json
print("json loaded")
import chromadb
print("chromadb loaded")
from sentence_transformers import SentenceTransformer
print("sentence transformer loaded")

# =========================
# CONFIG
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
DATA_FOLDER = "./data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "shakespeare"

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# INIT CHROMADB
# =========================

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

documents = []
metadatas = []
ids = []

# =========================
# PROCESS FILES
# =========================

for filename in os.listdir(DATA_FOLDER):

    # Only process scene chunk jsonl files
    if not filename.endswith(".jsonl"):
        continue

    if "scene_chunks" not in filename:
        continue

    print(f"\nProcessing {filename}...")

    filepath = os.path.join(DATA_FOLDER, filename)

    with open(filepath, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f):

            if not line.strip():
                continue

            item = json.loads(line)

            # =========================
            # EXTRACT TEXT
            # =========================

            text = item.get("text", "")

            if not text.strip():
                continue

            # =========================
            # CREATE ID
            # =========================

            chunk_id = (
                item.get("chunk_id")
                or item.get("scene_id")
                or f"{filename}_{line_number}"
            )

            # =========================
            # METADATA
            # =========================

            metadata = {
                "play": item.get("play"),
                "act": item.get("act"),
                "scene": item.get("scene"),
                "scene_id": item.get("scene_id"),
                "summary": item.get("scene_summary"),
                "keywords": ", ".join(
                    item.get("keywords", [])
                ),
                "source_file": filename
            }

            documents.append(text)
            metadatas.append(metadata)
            ids.append(chunk_id)

# =========================
# CREATE EMBEDDINGS
# =========================

print("\nGenerating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
).tolist()

# =========================
# STORE IN CHROMADB
# =========================

print("\nStoring in ChromaDB...")

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("\nDone!")
print(f"Stored {len(documents)} chunks.")