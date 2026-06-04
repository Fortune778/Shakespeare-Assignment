import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("--- BUILDING RICH SHAKESPEARE INDEX (WITH METADATA INJECTION) ---")

input_files = [
    "hamlet_scene_chunks.jsonl", 
    "macbeth_scene_chunks.jsonl", 
    "romeo_and_juliet_scene_chunks.jsonl"
]

# DENSER CHUNKING: 1000 characters with 300 overlap for high precision
TARGET_SIZE = 1000 
OVERLAP = 300      

texts_to_embed = []
metadata = []

for file_name in input_files:
    if not os.path.exists(file_name):
        print(f"⚠️ Warning: Missing {file_name}")
        continue
        
    print(f"Processing {file_name}...")
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue # Skip empty lines
            
            scene_data = json.loads(line)
            
            # --- 1. EXTRACT METADATA ---
            play_name = scene_data.get("play", "Unknown")
            act_num = scene_data.get("act", "Unknown")
            scene_num = scene_data.get("scene", "Unknown")
            
            # Grab the instructor's cheat codes!
            summary = scene_data.get("scene_summary", "")
            
            # Handle keywords properly (in case it's a list or string)
            keywords_list = scene_data.get("keywords", [])
            keywords = ", ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)
            
            # Build the Rich Header
            rich_header = f"[PLAY: {play_name} | ACT: {act_num} | SCENE: {scene_num}]\n[SUMMARY: {summary}]\n[KEYWORDS: {keywords}]\n---\n"
            
            full_text = scene_data.get("text", "")
            
            # --- 2. CHUNKING & INJECTION ---
            for i in range(0, len(full_text), TARGET_SIZE - OVERLAP):
                raw_chunk = full_text[i:i + TARGET_SIZE]
                if len(raw_chunk) < 200: continue

                # Glue the header to the text FOR THE EMBEDDER ONLY
                text_for_faiss = rich_header + raw_chunk

                # Append the enriched text to be vectorized by FAISS
                texts_to_embed.append(text_for_faiss)
                
                # Keep the clean text for the LLM to read later
                metadata.append({
                    "play": play_name,
                    "act": act_num,
                    "scene": scene_num,
                    "chunk_text": raw_chunk 
                })

print(f"Created {len(texts_to_embed)} highly-enriched chunks.")
print("Embedding chunks (this might take a minute)...")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# We encode the enriched text (texts_to_embed) instead of the raw text
embeddings = embedder.encode(texts_to_embed, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "data/preprocessed/shakespeare_master.index")
with open("master_metadata.json", "w", encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)

print("✅ Official Instructor Database Ready.")
