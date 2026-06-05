# Shakespeare-Assignment Installation and Environment Setup Guide
This project utilizes a conditionally routed RAG pipeline with a localized Small Language Model (Llama-3.2-3B-Instruct) loaded in 4-bit quantization. Due to the strict hardware constraints and the use of Low-Rank Adaptation (LoRA), please follow these setup instructions carefully to ensure the environment is configured correctly.
### 1. Prerequisites
 * **Operating System:** Windows or Linux (macOS is supported for CPU/MPS, but bitsandbytes 4-bit quantization is natively optimized for NVIDIA GPUs).
 * **Package Manager:** Anaconda or Miniconda is highly recommended to isolate dependencies.
 * **Hardware:** An NVIDIA GPU with at least 4GB of VRAM (e.g., RTX 2050 or higher).
 * **Hugging Face Account:** You must have a Hugging Face account and have accepted the user agreement for meta-llama/Llama-3.2-3B-Instruct to download the base weights.
### 2. Hardware and Driver Verification
Before building the environment, verify that your NVIDIA drivers are communicating correctly with the system. Open your terminal or command prompt and run:
```bash
nvidia-smi

```
*Note: Ensure the output displays your GPU details and the CUDA version. If this command fails, you need to update or reinstall your NVIDIA graphics drivers.*
### 3. Creating the Virtual Environment
To prevent dependency conflicts, create a fresh Anaconda environment. We will name this environment shakespeare_rag and use Python 3.10, which offers the best stability for the PyTorch and Hugging Face ecosystem.
```bash
# Create the fresh environment
conda create -n shakespeare_rag python=3.10 -y

# Activate the environment
conda activate shakespeare_rag

```
### 4. Installing PyTorch
You must install a version of PyTorch compiled with CUDA support. Run the following command to install PyTorch 2.0+ with CUDA 11.8 (or adjust the CUDA version to match your nvidia-smi output):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```
### 5. Installing Core Dependencies
With PyTorch installed, install the specific versions of the libraries required for RAG, vector retrieval, and quantized LLM execution.
Run the following command:
```bash
pip install transformers accelerate bitsandbytes peft sentence-transformers faiss-cpu numpy

```
 * bitsandbytes: Enables the 4-bit NF4 quantization to fit the 3B model into 4GB VRAM.
 * peft: Handles the QLoRA adapter routing.
 * faiss-cpu: Manages the vector similarity search.
 * sentence-transformers: Encodes the queries for semantic retrieval.
### 6. Hugging Face Authentication
Because Llama 3.2 is a gated model, you must authenticate your terminal with Hugging Face so the script can download the weights during its first run.
 1. Generate an Access Token in your Hugging Face account settings (ensure it has "Read" permissions).
 2. Install the Hugging Face Hub CLI (if not already installed via transformers):
   ```bash
   pip install huggingface_hub
   
   ```
 3. Login via the terminal:
   ```bash
   huggingface-cli login
   
   ```
 4. Paste your token when prompted.
### 7. Execution Instructions
Once the environment shakespeare_rag is active and dependencies are installed, you can run the system.

**Process the data first and store it into FAISS:**
```bash
python src/build_index.py

```
**To run the full pipeline (RAG + conditionally routed LoRA):**
```bash
python src/rag_lora.py

```
**To run the baseline RAG implementation (without style transfer):**
```bash
python src/rag_chatbot.py

```
**To evaluate the system using the provided test set:**
```bash
python evaluate/evaluate.py

```
The script will automatically detect the best available hardware (CUDA), load the model in 4-bit precision, and initialize a while True: listening loop for your queries.
