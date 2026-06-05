import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- TOP HALF: SETUP & LOADING ---
model_id = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading Llama 3.2 1B...")

# This is the line your code was missing! It defines the 'tokenizer'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# This loads the brain onto your 4GB RTX card
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto" 
)


# --- BOTTOM HALF: GENERATING THE RESPONSE ---

# 1. The Chat Template Structure
messages = [
    {"role": "system", "content": "You are William Shakespeare. You speak only in beautiful, authentic 16th-century Early Modern English."},
    {"role": "user", "content": "Say 'Hello World'."}
]

# 2. Apply the template to get the RAW STRING
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False, 
    add_generation_prompt=True
)

# 3. Tokenize the string into proper PyTorch tensors
inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

print("\nGenerating response... \n")

# 4. Generate Response
outputs = model.generate(
    **inputs, 
    max_new_tokens=100,
    temperature=0.7 
)

# 5. Strip the prompt out of the output
prompt_length = inputs['input_ids'].shape[-1]
response_ids = outputs[0][prompt_length:]

# 6. Decode and Print
print("--- Model Response ---")
print(tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
