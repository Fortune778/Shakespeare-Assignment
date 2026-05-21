import torch
import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

print("--- INITIALIZING SHAKESPEARE RAG (UNIVERSAL SCHOLAR EDITION) ---")

def get_optimal_device():
    """Detects and returns the best available hardware device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_optimal_device()
print(f"[*] Detected optimal hardware device: {device.upper()}")

# 1. Load Retriever
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 2. Configure & Load Base Model
print("[Loading Llama 3.2 3B...]")
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)

if device == "cuda":
    print("    -> Applying 4-bit quantization for CUDA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
else:
    print(f"    -> Loading natively in 16-bit for {device.upper()}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=device,
        torch_dtype=torch.bfloat16
    )

# 3. Load Database
index = faiss.read_index("shakespeare_master.index")
with open("master_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("System Ready!\n")

while True:
    user_query = input("\nYour Question: ")
    if user_query.lower() in ['quit', 'exit']: break

    # ==========================================
    # --- STEP 1: MULTI-QUERY EXPANSION ---
    # ==========================================
    print("\n[Agent 0: Generating Search Variants...]")
    
    expansion_messages = [
        {"role": "system", "content": "You are an expert search assistant. Generate 3 different search queries to help find the underlying motive, psychological reasons, or hidden context for the user's question. Output ONLY the queries, one per line. Do not number them."},
        {"role": "user", "content": user_query}
    ]
    
    exp_prompt = tokenizer.apply_chat_template(expansion_messages, tokenize=False, add_generation_prompt=True)
    exp_inputs = tokenizer(exp_prompt, return_tensors="pt").to(model.device)
    
    exp_outputs = model.generate(**exp_inputs, max_new_tokens=50, do_sample=False)
    expanded_text = tokenizer.decode(exp_outputs[0][exp_inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    
    queries = [user_query] + [q.strip() for q in expanded_text.strip().split("\n") if q.strip()]
    
    print(" > Executing Searches For:")
    for q in queries:
        print(f"    - {q}")

    # ==========================================
    # --- STEP 2: THE MEGA-RETRIEVAL (CAPPED AT 3) ---
    # ==========================================
    all_indices = []
    
    for q in queries:
        q_vec = embedder.encode([q])
        distances, indices = index.search(np.array(q_vec), k=3) 
        all_indices.extend(indices[0])

    # Remove duplicates and keep ONLY THE TOP 3 to prevent timeline hallucination!
    unique_indices = list(set(all_indices))[:3]
    
    context_text = ""
    citations = []
    for idx in unique_indices:
        meta = metadata[idx]
        context_text += f"--- TEXT FROM {meta['play'].upper()} ---\n{meta['chunk_text']}\n\n"
        citations.append(f"{meta['play']} (Act {meta['act']}, Scene {meta['scene']})")

    # ==========================================
    # --- STEP 3: GENERATION (THE UNIVERSAL PROMPT) ---
    # ==========================================
    print("\n[Agent 1: Reading Context & Generating Answer...]")
    
    system_instruction = "You are a highly precise, factual assistant and a master of Shakespearean translation."
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {user_query}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. Extract a factual summary from the context to fully answer the question. Rely ONLY on the provided context. (50 to 75 WORDS).\n"
        f"2. Translate that summary into 16th-century Early Modern English (50 to 75 WORDS).\n"
        f"3. Do NOT write a poem. Use EXACTLY the format below.\n\n"
        f"--- EXAMPLE FORMAT ---\n"
        f"EXTRACT: [50-75 words]\n"
        f"TRANSLATE: [50-75 words]\n"
        f"----------------------\n\n"
        f"Now answer the user's Question using the Context provided.\n"
        f"EXTRACT:"
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # STRICT GREEDY DECODING
    outputs = model.generate(
        **inputs, 
        max_new_tokens=250,    # Allows for ~150 words total (Extract + Translate)
        do_sample=False,       
        num_beams=1,           
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    
    print(f"\nThe Bard Replies:\nEXTRACT:{response}\n")
    
    print("--- Sources ---")
    for cite in sorted(set(citations)): 
        print(f" * {cite}")
    print("-------------------------")
