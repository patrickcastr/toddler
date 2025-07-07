# Fine-Tuning with Axolotl for Document Processing

This guide outlines the steps to fine-tune  to create the "Baby Todd Energy Assistant" - a specialized document processing assistant.

## Step 1: Create a Dataset Collection Tool

First, let's create a tool to collect example pairs of document processing prompts and their ideal responses.

Create the file: `c:\Users\user\LocalAIAssistant\tools\dataset_builder.py`

```python
import streamlit as st
import json
import os
from pathlib import Path

# Create the datasets folder if it doesn't exist
DATASET_DIR = Path(__file__).parent.parent / "datasets"
DATASET_DIR.mkdir(exist_ok=True)
DATASET_FILE = DATASET_DIR / "document_processing_examples.jsonl"

st.title("Baby Todd Energy Assistant Dataset Builder")

# Load existing dataset
examples = []
if DATASET_FILE.exists():
    with open(DATASET_FILE, "r") as f:
        examples = [json.loads(line) for line in f if line.strip()]

st.write(f"Current dataset contains {len(examples)} examples")

# Add new example
col1, col2 = st.columns(2)
with col1:
    st.subheader("Prompt")
    prompt = st.text_area("Enter the document processing request", height=100)
with col2:
    st.subheader("Assistant Response")
    response = st.text_area("Enter the ideal assistant response", height=300)

if st.button("Add Example") and prompt and response:
    example = {"prompt": prompt, "completion": response}
    examples.append(example)
    
    # Write to JSONL file
    with open(DATASET_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    st.success("Example added!")

# Display dataset
if examples:
    st.subheader("Current Dataset")
    for i, ex in enumerate(examples):
        with st.expander(f"Example {i+1}: {ex['prompt'][:30]}..."):
            st.write("**Prompt:**")
            st.write(ex["prompt"])
            st.write("**Response:**")
            st.code(ex["completion"], language="text")
```

Run this tool with `streamlit run tools/dataset_builder.py` and collect at least 20-30 examples of document processing tasks.

## Step 2: Prepare Your Environment

Install the necessary packages:

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install axolotl and its dependencies
pip install "axolotl[flash-attn,deepspeed]"
```

## Step 3: Create Axolotl Configuration File

Create the file: `c:\Users\user\LocalAIAssistant\tools\axolotl_config.yml`

```yaml
base_model: meta-llama/Llama-3-8B-Instruct
tokenizer_type: LlamaTokenizer
is_llama_derived_model: true

# Dataset configuration
datasets:
  - path: ../datasets/document_processing_examples.jsonl
    type: completion
    field_prompt: prompt
    field_completion: completion

# Model Configuration
model_config:
  trust_remote_code: true
  torch_dtype: bfloat16

# Training parameters
sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

gradient_accumulation_steps: 1
micro_batch_size: 1
num_epochs: 4
learning_rate: 2e-5
train_on_inputs: false
group_by_length: false
bf16: auto
fp16: auto
tf32: auto

# LoRA configuration
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - down_proj
  - up_proj

# Save directory
output_dir: ../models/baby_todd_energy_assistant
```

## Step 4: Run Fine-tuning

Execute the fine-tuning:

```bash
cd tools
axolotl train axolotl_config.yml
```

This process will take several hours depending on your GPU. For best results, use a machine with at least 16GB VRAM.

## Step 5: Convert to Ollama Format

After training completes, create a Modelfile to use with Ollama for the Baby Todd Energy Assistant:

