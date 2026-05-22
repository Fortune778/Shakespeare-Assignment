import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("--- LOADING THE SHAKESPEAREAN AI ---")

model_id = "meta-llama/Llama-3.2-3B-Instruct"

# 1. Load the Tokenizer (Warning Silenced!)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    clean_up_tokenization_spaces=False
)

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

# 2. Load the Base Model
print("Loading Base Model...")
if device == "cuda":
    print("    -> Applying 4-bit quantization for CUDA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
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

# 3. ATTACH THE LORA ADAPTER
print("Attaching your trained LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, "shakespeare_lora_final")

print("\n✅ AI is Ready! Type 'quit' to exit.\n")

# 4. The Chat Loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    # THE FIX: We MUST use the System Prompt we trained the LoRA with!
    messages = [
        {"role": "system", "content": "You are a master of 16th-century Shakespearean English."},
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # THE FIX: Increased max_new_tokens so it doesn't cut off!
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    print(f"\nAI: {response}\n")
