import sys
import os
import json
import faiss
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add assignment-2 directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assignment-2')))

from rag_lora import generate_rag_response, get_optimal_device

def main():
    print("--- STARTING ROUTING VERIFICATION TEST ---")
    
    # 1. Setup Environment
    device = get_optimal_device()
    print(f"[*] Optimal device: {device.upper()}")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("[1/3] Loading Model...")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    
    if device == "cuda":
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
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device,
            torch_dtype=torch.float16 if device == "mps" else torch.bfloat16
        )
    
    print("[2/3] Loading Shakespeare LoRA...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_2_dir = os.path.abspath(os.path.join(base_dir, "..", "assignment-2"))
    lora_path = os.path.join(assignment_2_dir, "shakespeare_lora_final")
    index_path = os.path.join(assignment_2_dir, "shakespeare_master.index")
    metadata_path = os.path.join(assignment_2_dir, "master_metadata.json")
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("[3/3] Loading DB...")
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    print("\nSystem ready! Testing queries:\n")
    
    # Query A: Conventional Question (should be Normal English)
    query_a = "Who is Macbeth and why is he important?"
    print(f"\n=======================")
    print(f"TEST A: '{query_a}'")
    print(f"=======================")
    _, response_a, _ = generate_rag_response(query_a, model, tokenizer, embedder, index, metadata)
    print(f"\nResponse A (Normal English expected):\n{response_a}\n")
    
    # Query B: Styled Question (should be Shakespearean)
    query_b = "Who is Macbeth in Shakespearean style?"
    print(f"\n=======================")
    print(f"TEST B: '{query_b}'")
    print(f"=======================")
    _, response_b, _ = generate_rag_response(query_b, model, tokenizer, embedder, index, metadata)
    print(f"\nResponse B (Early Modern English expected):\n{response_b}\n")
    
if __name__ == "__main__":
    main()
