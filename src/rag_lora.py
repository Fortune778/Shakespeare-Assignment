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
import re

def parse_query_metadata(query: str):
    """Parses a query to extract potential play, act, and scene filters."""
    query_lower = query.lower()
    
    # 1. Play detection
    plays = []
    if "hamlet" in query_lower:
        plays.append("Hamlet")
    if "macbeth" in query_lower:
        plays.append("Macbeth")
    if "romeo" in query_lower or "juliet" in query_lower:
        plays.append("Romeo and Juliet")
        
    # 2. Act & Scene detection (supports 'act 1', 'scene 3', roman numerals like 'act i', 'scene iii')
    act = None
    scene = None
    
    act_match = re.search(r"\bact\s+(\d+)\b", query_lower)
    if act_match:
        act = int(act_match.group(1))
    else:
        # Check Roman numerals (I to V)
        roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
        act_match = re.search(r"\bact\s+([ivx]+)\b", query_lower)
        if act_match and act_match.group(1) in roman_map:
            act = roman_map[act_match.group(1)]
            
    scene_match = re.search(r"\bscene\s+(\d+)\b", query_lower)
    if scene_match:
        scene = int(scene_match.group(1))
    else:
        # Check Roman numerals (I to V)
        roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
        scene_match = re.search(r"\bscene\s+([ivx]+)\b", query_lower)
        if scene_match and scene_match.group(1) in roman_map:
            scene = roman_map[scene_match.group(1)]
            
    return plays, act, scene


def detect_shakespeare_request(query: str) -> bool:
    """Detects if the user explicitly asked for Shakespearean style or 16th-century English."""
    keywords = [
        "shakespeare", "early modern english", "16th-century", "16th century", 
        "bard", "old english", "poetic", "verse", "verily", "forsooth", 
        "thou", "thee", "thy", "translate", "style of"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)


def generate_rag_response(user_query, model, tokenizer, embedder, index, metadata):
    # Detect style request
    wants_shakespeare = detect_shakespeare_request(user_query)

    # ==========================================
    # --- STEP 1: QUERY EXPANSION (NO ROUTING) ---
    # ==========================================
    print("\n[Agent 0: Expanding Search Parameters...]")
    
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
    
    # Run expansion with base model (LoRA disabled) for factual clarity
    with model.disable_adapter():
        exp_outputs = model.generate(**exp_inputs, max_new_tokens=40, do_sample=False)
    expanded_text = tokenizer.decode(exp_outputs[0][exp_inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    
    queries = [user_query] + [q.strip() for q in expanded_text.split('\n') if q.strip()]

    # ==========================================
    # --- STEP 2: HYBRID METADATA-AWARE RETRIEVAL ---
    # ==========================================
    plays_filter, act_filter, scene_filter = parse_query_metadata(user_query)
    
    # Direct Act/Scene matching for highly specific queries
    direct_indices = []
    if plays_filter and len(plays_filter) == 1 and act_filter is not None and scene_filter is not None:
        target_play = plays_filter[0]
        for idx, meta in enumerate(metadata):
            if meta['play'] == target_play and meta['act'] == act_filter and meta['scene'] == scene_filter:
                direct_indices.append(idx)
                
    if direct_indices:
        print(f" -> Found direct metadata matches for {plays_filter[0]} Act {act_filter}, Scene {scene_filter}!")
        unique_indices = direct_indices[:5]
    else:
        # Run semantic vector search
        all_indices = []
        for q in queries:
            q_vec = embedder.encode([q])
            # Retrieve more candidates to allow for successful play filtering
            distances, indices = index.search(np.array(q_vec), k=12) 
            all_indices.extend(indices[0])
            
        # Filter retrieved indices by plays if specific play filters are mentioned in the query
        if plays_filter:
            filtered_indices = []
            for idx in all_indices:
                if metadata[idx]['play'] in plays_filter:
                    filtered_indices.append(idx)
            # Use filtered list only if we have sufficient context chunks left
            if len(filtered_indices) >= 2:
                all_indices = filtered_indices
                
        unique_indices = []
        for idx in all_indices:
            if idx not in unique_indices:
                unique_indices.append(idx)
        unique_indices = unique_indices[:5]
        
    # Sort chronologically (Act -> Scene) to preserve the play's natural dramatic flow
    final_indices = sorted(unique_indices, key=lambda i: (metadata[i]['act'], metadata[i]['scene']))
    
    context_text = ""
    citations = []
    for idx in final_indices:
        meta = metadata[idx]
        context_text += f"--- {meta['play'].upper()} (ACT {meta['act']}, SCENE {meta['scene']}) ---\n{meta['chunk_text']}\n\n"
        citations.append(f"{meta['play']} (Act {meta['act']}, Scene {meta['scene']})")

    # ==========================================
    # --- CONDITIONAL RESPONSE ROUTING ---
    # ==========================================
    if wants_shakespeare:
        # ==========================================
        # --- PASS 1: BULLET-POINT EXTRACTION ---
        # ==========================================
        print("[Agent 1: Extracting Raw Facts (LoRA Disabled for Accuracy)...]")
        
        fact_instruction = (
            "You are a strict data extractor. Read the context and answer the question using ONLY facts explicitly stated in the text.\n"
            "CRITICAL:\n"
            "1. Rely ONLY on the provided context. Do not make up facts, relationships, or use external knowledge. Do not assume or extrapolate.\n"
            "2. If the context does not explicitly contain the answer, output 'Fact not mentioned in context.'\n"
            "3. Output exactly 2 or 3 short bullet points. Do not write a paragraph."
        )
        
        fact_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nFACTS:\n-"
        
        msg_fact = [
            {"role": "system", "content": fact_instruction},
            {"role": "user", "content": fact_prompt}
        ]
        
        p1_text = tokenizer.apply_chat_template(msg_fact, tokenize=False, add_generation_prompt=True)
        p1_inputs = tokenizer(p1_text, return_tensors="pt").to(model.device)
        
        # Run fact extraction with LoRA disabled to get highly accurate factual summary
        with model.disable_adapter():
            p1_outputs = model.generate(**p1_inputs, max_new_tokens=100, do_sample=False) 
        factual_summary = tokenizer.decode(p1_outputs[0][p1_inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

        # ==========================================
        # --- PASS 2: EARLY MODERN PROSE TRANSLATION ---
        # ==========================================
        print("[Agent 2: Translating into 16th-Century Prose (LoRA Active)...]")
        
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

        # Run translation with LoRA enabled to get authentic Shakespearean style
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
        
        if "Note:" in translation:
            translation = translation.split("Note:")[0].strip()
        if "Here is" in translation:
            translation = translation.split("Here is")[0].strip()

    else:
        # ==========================================
        # --- STANDARD RESPONSE (NORMAL ENGLISH) ---
        # ==========================================
        # We leverage a two-pass system here to eliminate RAG degradation!
        # Pass 1: Extract direct modern factual statements from the retrieved text.
        # Pass 2: Synthesize those clear facts into a natural modern English paragraph.
        
        # --- PASS 1: BULLET-POINT EXTRACTION ---
        print("[Agent 1: Extracting Modern Facts from Dialogue (LoRA Disabled for Accuracy)...]")
        
        modern_fact_instruction = (
            "You are a precise literary data extractor. Read the provided Shakespearean context and extract ONLY the direct, modern factual statements relevant to the question.\n"
            "CRITICAL RULES:\n"
            "1. Simplify complex Shakespearean dialogue or soliloquies into direct, plain modern facts.\n"
            "2. Under no circumstances should you swap character identities, roles, or relationships. Make sure you correctly identify who does what (e.g. Macbeth is the Thane/King, Macbeth kills Duncan, Banquo is Macbeth's friend and fellow general, Romeo buys the poison from the apothecary, etc.).\n"
            "3. If the context does not contain the answer, output 'Fact not mentioned in context.'\n"
            "4. Output exactly 2 or 3 short bullet points. Do not write a paragraph."
        )
        
        modern_fact_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nFACTS:\n-"
        
        msg_fact = [
            {"role": "system", "content": modern_fact_instruction},
            {"role": "user", "content": modern_fact_prompt}
        ]
        
        p1_text = tokenizer.apply_chat_template(msg_fact, tokenize=False, add_generation_prompt=True)
        p1_inputs = tokenizer(p1_text, return_tensors="pt").to(model.device)
        
        with model.disable_adapter():
            p1_outputs = model.generate(**p1_inputs, max_new_tokens=120, do_sample=False)
        factual_summary = tokenizer.decode(p1_outputs[0][p1_inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        print(f" -> Extracted Facts:\n- {factual_summary}")
        
        # --- PASS 2: MODERN PARAGRAPH SYNTHESIS ---
        print("[Agent 2: Synthesizing natural modern English response...]")
        
        modern_synthesis_instruction = (
            "You are a helpful, precise, and academic assistant. Synthesize the provided facts into a clear, natural modern English response.\n"
            "CRITICAL RULES:\n"
            "1. STRICT FACTUAL LIMITATION: Rely ONLY on the provided facts. Do not make up any details, assumptions, or external relationships. If the facts state the detail is not mentioned, say so.\n"
            "2. Write in normal, clean modern English. Do NOT use any archaic words or Shakespearean dialogue.\n"
            "3. Write a single, well-structured, cohesive paragraph (50 to 100 words)."
        )
        
        modern_synthesis_prompt = f"Modern Facts:\n- {factual_summary}\n\nResponse:\n"
        
        msg_synthesis = [
            {"role": "system", "content": modern_synthesis_instruction},
            {"role": "user", "content": modern_synthesis_prompt}
        ]
        
        p2_text = tokenizer.apply_chat_template(msg_synthesis, tokenize=False, add_generation_prompt=True)
        p2_inputs = tokenizer(p2_text, return_tensors="pt").to(model.device)
        
        with model.disable_adapter():
            p2_outputs = model.generate(
                **p2_inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        translation = tokenizer.decode(p2_outputs[0][p2_inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

    return context_text, translation, citations


def get_optimal_device():
    """Detects and returns the best available hardware device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    print("--- INITIALIZING SHAKESPEARE RAG (PROSE EDITION) ---")
    
    device = get_optimal_device()
    print(f"[*] Detected optimal hardware device: {device.upper()}")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("[1/3] Loading Base Model...")
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
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        print(f"    -> Loading natively in 16-bit for {device.upper()}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device,
            torch_dtype=torch.float16 if device == "mps" else torch.bfloat16
        )
    
    print("[2/3] Attaching Shakespeare LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, "shakespeare_lora_final")
    
    print("[3/3] Loading Database...")
    index = faiss.read_index("shakespeare_master.index")
    with open("master_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print("\n✅ System Ready! The Bard will now speak in prose.\n")
    
    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() in ['quit', 'exit']: break

        context_text, translation, citations = generate_rag_response(
            user_query, model, tokenizer, embedder, index, metadata
        )
        
        print(f"\n--- THE HISTORIAN REPLIES ---")
        print(f"{translation}")
        
        print("\n--- Sources ---")
        for cite in sorted(set(citations)): 
            print(f" * {cite}")
        print("-------------------------")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

