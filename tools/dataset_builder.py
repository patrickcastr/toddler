import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path

# Create the datasets folder if it doesn't exist
DATASET_DIR = Path(__file__).parent.parent / "datasets"
DATASET_DIR.mkdir(exist_ok=True)
DATASET_FILE = DATASET_DIR / "document_processing_examples.jsonl"

def main():
    st.title("Baby Todd Energy Assistant Dataset Builder")
    
    # Load existing dataset
    examples = []
    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r") as f:
            examples = [json.loads(line) for line in f if line.strip()]
    
    st.write(f"Current dataset contains {len(examples)} examples")
    
    # Input form
    with st.form("new_example"):
        st.subheader("Add New Example")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prompt = st.text_area("User Request (document processing task)", 
                                height=100, 
                                placeholder="e.g., Summarize this energy report in 3 bullet points")
            
        with col2:
            completion = st.text_area("Ideal Assistant Response", 
                                   height=300,
                                   placeholder="Here's a summary of the energy report:\n• Point 1...\n• Point 2...")
        
        # Custom tags for categorization
        tags = st.multiselect(
            "Tags (optional)",
            options=["Summary", "Analysis", "Extraction", "Translation", "Q&A", "Energy", "Reports", "Technical"],
            default=[]
        )
        
        submit = st.form_submit_button("Add to Dataset")
        
    if submit and prompt and completion:
        # Create example with metadata
        example = {
            "prompt": prompt,
            "completion": completion,
            "tags": tags
        }
        
        # Add to examples list
        examples.append(example)
        
        # Save to JSONL (only write prompt and completion)
        with open(DATASET_FILE, "w") as f:
            for ex in examples:
                # Create simplified version for training
                train_ex = {"prompt": ex["prompt"], "completion": ex["completion"]}
                f.write(json.dumps(train_ex) + "\n")
        
        st.success("Example added to dataset!")
        st.balloons()
    
    # Display dataset statistics
    if examples:
        st.subheader("Dataset Statistics")
        
        # Count by tags
        tag_counts = {}
        for ex in examples:
            for tag in ex.get("tags", []):
                if tag in tag_counts:
                    tag_counts[tag] += 1
                else:
                    tag_counts[tag] = 1
        
        # Create two columns for stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Examples", len(examples))
            
        with col2:
            if tag_counts:
                st.write("Tag Distribution:")
                for tag, count in tag_counts.items():
                    st.write(f"- {tag}: {count}")
            else:
                st.write("No tags used yet")
        
        # Display examples table
        st.subheader("Examples Overview")
        
        # Convert to DataFrame for display
        df = pd.DataFrame([{
            "Request": ex["prompt"][:50] + "..." if len(ex["prompt"]) > 50 else ex["prompt"],
            "Response Length": len(ex["completion"]),
            "Tags": ", ".join(ex.get("tags", []))
        } for ex in examples])
        
        st.dataframe(df)
        
        # Detailed view with expandable sections
        st.subheader("Detailed Examples")
        for i, ex in enumerate(examples):
            with st.expander(f"Example {i+1}: {ex['prompt'][:40]}..."):
                st.write("**Request:**")
                st.write(ex["prompt"])
                st.write("**Assistant Response:**")
                st.code(ex["completion"], language="text")
                if ex.get("tags"):
                    st.write("**Tags:**", ", ".join(ex["tags"]))
                
                # Add buttons to edit or delete
                col1, col2 = st.columns(2)
                with col2:
                    if st.button(f"Delete Example {i+1}"):
                        del examples[i]
                        
                        # Save updated dataset
                        with open(DATASET_FILE, "w") as f:
                            for remaining_ex in examples:
                                # Create simplified version for training
                                train_ex = {"prompt": remaining_ex["prompt"], "completion": remaining_ex["completion"]}
                                f.write(json.dumps(train_ex) + "\n")
                        
                        st.success("Example deleted! Refresh the page to see updated dataset.")
                        st.rerun()
        
        # Export options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download JSONL Dataset",
                data="\n".join([json.dumps({"prompt": ex["prompt"], "completion": ex["completion"]}) for ex in examples]),
                file_name="document_processing_examples.jsonl",
                mime="application/json"
            )
            
        with col2:
            st.download_button(
                label="Download Full Dataset with Tags (JSON)",
                data=json.dumps(examples, indent=2),
                file_name="document_processing_examples_with_metadata.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
