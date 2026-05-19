import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("--- BUILDING BALANCED SHAKESPEARE INDEX ---")

input_files = [
    "hamlet_scene_chunks.jsonl", 
    "macbeth_scene_chunks.jsonl", 
    "romeo_and_juliet_scene_chunks.jsonl"
]

# BALANCED SETTINGS: 2000 characters with 400 overlap
TARGET_SIZE = 1000 
OVERLAP = 300      

chunks_text = []
metadata = []

for file_name in input_files:
    if not os.path.exists(file_name):
        print(f"⚠️ Warning: Missing {file_name}")
        continue
        
    print(f"Processing {file_name}...")
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            scene_data = json.loads(line)
            full_text = scene_data.get("text", "")
            
            for i in range(0, len(full_text), TARGET_SIZE - OVERLAP):
                chunk_text = full_text[i:i + TARGET_SIZE]
                if len(chunk_text) < 200: continue

                chunks_text.append(chunk_text)
                metadata.append({
                    "play": scene_data.get("play", "Unknown"),
                    "act": scene_data.get("act", "Unknown"),
                    "scene": scene_data.get("scene", "Unknown"),
                    "chunk_text": chunk_text
                })

print(f"Created {len(chunks_text)} balanced chunks.")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks_text, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "shakespeare_master.index")
with open("master_metadata.json", "w", encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)

print("✅ Balanced Database Ready.")
