"""
Tool for collecting and preparing fine-tuning datasets for OpenSCAD model generation
"""

import os
import json
import pandas as pd
from pathlib import Path
import streamlit as st

# Define the dataset file path
DATASET_DIR = Path(__file__).parent.parent / "datasets"
DATASET_FILE = DATASET_DIR / "openscad_examples.jsonl"

def main():
    st.title("OpenSCAD Fine-tuning Dataset Builder")
    
    # Make sure the dataset directory exists
    DATASET_DIR.mkdir(exist_ok=True)
    
    # Initialize the dataset if it doesn't exist
    if not DATASET_FILE.exists():
        with open(DATASET_FILE, "w") as f:
            pass  # Create an empty file
    
    # Load existing dataset
    examples = []
    if DATASET_FILE.exists() and DATASET_FILE.stat().st_size > 0:
        with open(DATASET_FILE, "r") as f:
            examples = [json.loads(line) for line in f if line.strip()]
    
    # Display stats
    st.write(f"Current dataset contains {len(examples)} examples")
    
    # Interface for adding new examples
    st.header("Add New Example")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prompt")
        prompt = st.text_area("Enter the description of the 3D model", height=100)
        
    with col2:
        st.subheader("Expected OpenSCAD Code")
        completion = st.text_area("Enter the correct OpenSCAD code for this prompt", height=300)
    
    if st.button("Add to Dataset") and prompt and completion:
        # Add the example
        example = {"prompt": prompt, "completion": completion}
        examples.append(example)
        
        # Write the dataset
        with open(DATASET_FILE, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        
        st.success("Example added to dataset!")
    
    # Show dataset
    if examples:
        st.header("Current Dataset")
        df = pd.DataFrame(examples)
        st.dataframe(df)
        
        # Export options
        st.download_button(
            "Download JSONL Dataset",
            "\n".join([json.dumps(ex) for ex in examples]),
            "openscad_examples.jsonl",
            "text/plain"
        )
    
    # Fine-tuning instructions
    with st.expander("Fine-tuning Instructions"):
        st.markdown("""
        ## Fine-Tuning with Axolotl
        
        1. **Install Axolotl**:
        ```bash
        pip install axolotl
        ```
        
        2. **Create a config file** (config.yml):
        ```yaml
        # For Document Processing Model
        base_model: llamafactory/codellama:7b
        model_type: LlamaForCausalLM
        tokenizer_type: LlamaTokenizer
        is_llama_derived_model: true
        
        sequence_len: 4096  # Increased for document processing
        sample_packing: true
        
        datasets:
          - path: document_processing_examples.jsonl
            type: completion
        
        lora_r: 8
        lora_alpha: 16
        lora_dropout: 0.05
        lora_target_modules:
          - q_proj
          - k_proj
          - v_proj
          - o_proj
          - gate_proj  # Added for better text processing
        
        output_dir: ./fine_tuned_doc_processor
        ```
        
        3. **Run fine-tuning**:
        ```bash
        python -m axolotl.cli.train config.yml
        ```
        
        > **Note**: Fine-tuning requires significant GPU resources (at least 24GB VRAM).
        > Consider using cloud options like Google Colab Pro or RunPod if you don't have a suitable GPU.
        """)

if __name__ == "__main__":
    main()
