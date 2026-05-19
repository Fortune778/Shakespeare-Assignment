import os
# Force PyTorch to prevent memory fragmentation on small GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

print("--- IGNITING THE FORGE: QLoRA SHAKESPEARE TRAINING (NATIVE BFLOAT16 EDITION) ---")

# ==========================================
# 1. LOAD THE UTTERANCE DATASETS
# ==========================================
print("\n[1/6] Loading Shakespearean Utterances...")
data_files = [
    "hamlet_utterances.jsonl", 
    "macbeth_utterances.jsonl", 
    "romeo_and_juliet_utterances.jsonl"
]

dataset = load_dataset("json", data_files=data_files, split="train")
dataset = dataset.filter(lambda x: x["speaker"] != "STAGE_DIRECTION")

# ==========================================
# 2. LOAD TOKENIZER EARLY
# ==========================================
print("\n[2/6] Loading Tokenizer...")
model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

# ==========================================
# 3. FORMAT & MANUALLY TRUNCATE DATA
# ==========================================
print("\n[3/6] Formatting and Truncating Dataset...")
def format_instruction(example):
    system_prompt = "You are a master of 16th-century Shakespearean English."
    user_prompt = f"Speak a line in the style of {example['speaker']} from {example['play']}."
    assistant_response = example['text']
    
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>"
    )
    
    # MANUAL BYPASS: Force the string to be <= 128 tokens
    tokens = tokenizer(text, truncation=True, max_length=128)
    truncated_text = tokenizer.decode(tokens["input_ids"], skip_special_tokens=False)
    
    return {"text": truncated_text}

dataset = dataset.map(format_instruction)

# ==========================================
# 4. LOAD THE BASE MODEL (THE 4-BIT TEXTBOOK)
# ==========================================
print("\n[4/6] Loading Base Model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # <--- CHANGED: Native BFloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16            # <--- CHANGED: Native BFloat16
)

# THE CUSTOM 4GB VRAM HACK: BYPASS THE 1.5GB UPCAST
print("      Applying Surgical VRAM Hack...")
for name, param in model.named_parameters():
    param.requires_grad = False
    if param.ndim == 1 and "norm" in name.lower():
        param.data = param.data.to(torch.float32)

model.enable_input_require_grads() 
model.gradient_checkpointing_enable()

# ==========================================
# 5. CONFIGURE THE LORA ADAPTER
# ==========================================
print("\n[5/6] Configuring the 16-bit LoRA Adapter...")
peft_config = LoraConfig(
    r=8,                  
    lora_alpha=16,        
    lora_dropout=0.05,    
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] 
)

# ==========================================
# 6. CONFIGURE & RUN THE TRAINING 
# ==========================================
print("\n[6/6] Configuring the Training Engine...")
training_args = SFTConfig(
    output_dir="./shakespeare_lora_checkpoints",
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=4,       
    optim="paged_adamw_32bit",           
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,                          # <--- TURNED OFF
    bf16=True,                           # <--- TURNED ON (RTX 2050 SUPPORTS THIS!)
    max_steps=100,                       
    save_steps=50,
    report_to="none",
    dataset_text_field="text"            
)

trainer = SFTTrainer(
    model=model,                         
    train_dataset=dataset,
    peft_config=peft_config,             
    processing_class=tokenizer,          
    args=training_args,                  
)

print("\n🔥 THE FORGE IS BURNING! Starting Training...")
trainer.train()

# ==========================================
# 7. SAVE THE ADAPTER
# ==========================================
print("\n✅ Training Complete! Saving your LoRA Adapter...")
trainer.model.save_pretrained("shakespeare_lora_final")
tokenizer.save_pretrained("shakespeare_lora_final")
print("All done. The 'shakespeare_lora_final' folder is ready for your Chatbot.")
