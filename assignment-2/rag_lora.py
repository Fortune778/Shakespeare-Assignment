import os
# Essential for running complex RAG on 4GB VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from peft import PeftModel

print("--- INITIALIZING SHAKESPEARE RAG (PROSE EDITION) ---")

# 1. Load Retriever
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Configure 4-bit Quantization
print("[1/3] Loading Base Model into VRAM...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 3. ATTACH THE LORA ADAPTER
print("[2/3] Attaching Shakespeare LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, "shakespeare_lora_final")

# 4. Load Database
print("[3/3] Loading Database...")
index = faiss.read_index("shakespeare_master.index")
with open("master_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("\n✅ System Ready! The Bard will now speak in prose.\n")

while True:
    user_query = input("\nYour Question: ")
    if user_query.lower() in ['quit', 'exit']: break

    # ==========================================
    # --- STEP 1: QUERY EXPANSION (NO ROUTING) ---
    # ==========================================
    print("\n[Agent 0: Expanding Search Parameters...]")
    
    # We still expand the query to get better database hits, but we don't force a play name.
    expansion_instruction = (
        "You are a search assistant. Generate 2 alternative search queries to help find the facts or motives behind the user's question.\n"
        "Output ONLY the queries, one per line. Do not number them."
    )
    
    expansion_messages = [
        {"role": "system", "content": expansion_instruction},
        {"role": "user", "content": user_query}
    ]
    
    exp_prompt = tokenizer.apply_chat_template(expansion_messages, tokenize=False, add_generation_prompt=True)
    exp_inputs = tokenizer(exp_prompt, return_tensors="pt").to(model.device)
    exp_outputs = model.generate(**exp_inputs, max_new_tokens=40, do_sample=False)
    expanded_text = tokenizer.decode(exp_outputs[0][exp_inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    
    queries = [user_query] + [q.strip() for q in expanded_text.split('\n') if q.strip()]

    # ==========================================
    # --- STEP 2: CHRONOLOGICAL RETRIEVAL ---
    # ==========================================
    all_indices = []
    for q in queries:
        q_vec = embedder.encode([q])
        distances, indices = index.search(np.array(q_vec), k=3) 
        all_indices.extend(indices[0])

    # Remove duplicates and keep the top 5 chunks
    unique_indices = list(set(all_indices))[:5]
    
    # Sort chronologically (Act -> Scene) to preserve the timeline
    final_indices = sorted(unique_indices, key=lambda i: (metadata[i]['act'], metadata[i]['scene']))
    
    context_text = ""
    citations = []
    for idx in final_indices:
        meta = metadata[idx]
        context_text += f"--- {meta['play'].upper()} (ACT {meta['act']}, SCENE {meta['scene']}) ---\n{meta['chunk_text']}\n\n"
        citations.append(f"{meta['play']} (Act {meta['act']}, Scene {meta['scene']})")

    # ==========================================
    # --- PASS 1: BULLET-POINT EXTRACTION ---
    # ==========================================
    print("[Agent 1: Extracting Raw Facts...]")
    
    # We keep bullet points because it stops 3B models from hallucinating causality.
    fact_instruction = (
        "You are a strict data extractor. Read the context and answer the question using ONLY facts found in the text.\n"
        "CRITICAL: Output exactly 2 or 3 short bullet points. Do not write a paragraph."
    )
    
    fact_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nFACTS:\n-"
    
    msg_fact = [
        {"role": "system", "content": fact_instruction},
        {"role": "user", "content": fact_prompt}
    ]
    
    p1_text = tokenizer.apply_chat_template(msg_fact, tokenize=False, add_generation_prompt=True)
    p1_inputs = tokenizer(p1_text, return_tensors="pt").to(model.device)
    
    p1_outputs = model.generate(**p1_inputs, max_new_tokens=100, do_sample=False) 
    factual_summary = tokenizer.decode(p1_outputs[0][p1_inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

    # ==========================================
    # --- PASS 2: EARLY MODERN PROSE TRANSLATION ---
    # ==========================================
    print("[Agent 2: Translating into 16th-Century Prose...]")
    
    # PROMPT ENG: Explicitly forbid poetry/verse and demand continuous prose.
    style_instruction = (
        "You are a 16th-century historian. Translate the provided modern facts into Early Modern English.\n"
        "CRITICAL RULES:\n"
        "1. Write in continuous PROSE (like a letter or chronicle). Do NOT write poetry. Do NOT use line breaks or verse.\n"
        "2. Preserve the factual meaning exactly.\n"
        "3. Output ONLY the translated prose. No introductory notes."
    )
    
    style_prompt = f"Modern Facts:\n- {factual_summary}\n\nSHAKESPEAREAN PROSE TRANSLATION:\n"
    
    msg_style = [
        {"role": "system", "content": style_instruction},
        {"role": "user", "content": style_prompt}
    ]

    p2_text = tokenizer.apply_chat_template(msg_style, tokenize=False, add_generation_prompt=True)
    p2_inputs = tokenizer(p2_text, return_tensors="pt").to(model.device)

    # Increased max_tokens to 200 for paragraph generation; slightly higher temperature for vocabulary variety
    p2_outputs = model.generate(
        **p2_inputs, 
        max_new_tokens=200,
        do_sample=True,        
        temperature=0.3, 
        top_p=0.85,
        repetition_penalty=1.15, 
        pad_token_id=tokenizer.eos_token_id
    )

    translation = tokenizer.decode(p2_outputs[0][p2_inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    
    # Post-Process Scrubber to catch any stubborn AI chat habits
    if "Note:" in translation:
        translation = translation.split("Note:")[0].strip()
    if "Here is" in translation:
        translation = translation.split("Here is")[0].strip()
    
    # ==========================================
    # --- FINAL OUTPUT ---
    # ==========================================
    print(f"\n--- THE HISTORIAN REPLIES ---")
    print(f"{translation}")
    
    print("\n--- Sources ---")
    for cite in sorted(set(citations)): 
        print(f" * {cite}")
    print("-------------------------")

    torch.cuda.empty_cache()
