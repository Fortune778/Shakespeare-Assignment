import csv
import json
import os
import sys

import faiss
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add the assignment-2 directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assignment-2')))

try:
    from rag_lora import generate_rag_response
except ImportError as e:
    print(f"Warning: Could not import assignment-2 modules. Details: {e}")
    generate_rag_response = None


def get_baseline_response(question, model, tokenizer):
    """
    Wrapper for the Baseline System.
    Calls the SLM directly without retrieved context.
    We temporarily disable the LoRA adapter to get the pure base model response.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question clearly and factually."},
        {"role": "user", "content": question}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with model.disable_adapter():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return response


def get_rag_response(question, model, tokenizer, embedder, index, metadata):
    """
    Wrapper for the Improved RAG System.
    """
    if generate_rag_response:
        context_text, translation, citations = generate_rag_response(
            question, model, tokenizer, embedder, index, metadata
        )
        return context_text, translation
    return ("Retrieved passage placeholder.", "RAG response placeholder.")


def get_optimal_device():
    """Detects and returns the best available hardware device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    # Set working directory to assignment-2 to load index and metadata correctly
    original_cwd = os.getcwd()
    assignment_2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assignment-2'))
    os.chdir(assignment_2_dir)

    print("--- INITIALIZING EVALUATION SYSTEM ---")
    
    # Detect and print optimal hardware
    device = get_optimal_device()
    print(f"[*] Detected optimal hardware device: {device.upper()}")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("[1/3] Loading Base Model...")
    
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    
    if device == "cuda":
        # Apply 4-bit quantization for Nvidia/AMD GPUs to save VRAM
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
            dtype=torch.bfloat16
        )
    else:
        # Load natively for Apple Silicon (MPS) or CPU
        print(f"    -> Loading natively in 16-bit for {device.upper()}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device,
            dtype=torch.float16 if device == "mps" else torch.bfloat16
        )
    
    print("[2/3] Attaching Shakespeare LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, "shakespeare_lora_final")
    
    print("[3/3] Loading Database...")
    index = faiss.read_index("shakespeare_master.index")
    with open("master_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Revert to original directory
    os.chdir(original_cwd)

    # 1. Test Set Definition
    test_questions = [
        # Instructor-Provided Questions (From the specification):
        {
            "question": "Who is Hamlet?",
            "expected_focus": "Hamlet, character concept"
        },
        {
            "question": "What is the role of Lady Macbeth?",
            "expected_focus": "Macbeth, character concept"
        },
        {
            "question": "Why does Macbeth kill Duncan?",
            "expected_focus": "Macbeth, contextual motivation"
        },
        {
            "question": "Why does Hamlet delay taking revenge?",
            "expected_focus": "Hamlet, contextual motivation"
        },
        {
            "question": "Why is Juliet conflicted after meeting Romeo?",
            "expected_focus": "Romeo and Juliet, contextual motivation"
        },
        # Group-Designed Questions (For robustness, style, and edge cases):
        {
            "question": "What is the conflict between the Montagues and the Capulets?",
            "expected_focus": "Romeo and Juliet, family feud"
        },
        {
            "question": "Who is Banquo, and what is his relationship with Macbeth?",
            "expected_focus": "Macbeth, character relationship"
        },
        {
            "question": "What role does the poison play in the final scene of Romeo and Juliet?",
            "expected_focus": "Romeo and Juliet, plot details"
        },
        {
            "question": "Explain in a Shakespearean style (under 150 words): How does Macbeth feel after seeing the ghost?",
            "expected_focus": "Macbeth, style generation"
        },
        {
            "question": "Did Hamlet ever meet Juliet?",
            "expected_focus": "Robustness, hallucination check - they never met"
        },
        {
            "question": "What is the exact name of the poison Juliet takes?",
            "expected_focus": "Robustness, acknowledging lack of evidence in text"
        },
        {
            "question": "Explain the ending of Hamlet to an 8-year-old child.",
            "expected_focus": "Beginner usefulness"
        },
        {
            "question": "Summarize the events of Act 1, Scene 3 in Macbeth.",
            "expected_focus": "Testing scene-level retrieval capabilities"
        },
        {
            "question": "How do the ghosts in 'Hamlet' and 'Macbeth' differ in their purpose?",
            "expected_focus": "Cross-play comparison and synthesis"
        },
        {
            "question": "What does the phrase 'star-crossed lovers' mean in the context of Romeo and Juliet?",
            "expected_focus": "Explaining literary themes to beginners"
        }
    ]

    # 4. Output Formatting Setup
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_file_path = os.path.join(results_dir, 'evaluation_results.csv')
    fieldnames = [
        'question', 
        'expected_focus', 
        'retrieved_passage', 
        'baseline_response', 
        'rag_response', 
        'correctness_score', 
        'grounding_score', 
        'retrieval_relevance_score', 
        'usefulness_score', 
        'style_quality_score', 
        'comments'
    ]

    results = []
    
    print(f"\nStarting evaluation of {len(test_questions)} questions...")
    
    # 3. Evaluation Loop
    for idx, item in enumerate(test_questions, 1):
        question = item['question']
        print(f"\nProcessing Q{idx}: {question}")
        
        # Call the Baseline System
        print(" -> Generating Baseline Response...")
        baseline_response = get_baseline_response(question, model, tokenizer)
        
        # Call the Improved RAG System
        print(" -> Generating Improved RAG Response...")
        retrieved_passage, rag_response = get_rag_response(question, model, tokenizer, embedder, index, metadata)
        
        # Record all values, leaving scores and comments blank for manual entry
        result_row = {
            'question': question,
            'expected_focus': item['expected_focus'],
            'retrieved_passage': retrieved_passage,
            'baseline_response': baseline_response,
            'rag_response': rag_response,
            'correctness_score': '',
            'grounding_score': '',
            'retrieval_relevance_score': '',
            'usefulness_score': '',
            'style_quality_score': '',
            'comments': ''
        }
        results.append(result_row)

    # Export to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"\nEvaluation complete. Results saved to: {csv_file_path}")

if __name__ == "__main__":
    main()
