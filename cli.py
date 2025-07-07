#!/usr/bin/env python
"""
Command line interface for generating 3D models
"""
import sys
import os
import argparse
from api import generate_model
import inspect

def main():
    parser = argparse.ArgumentParser(description="Generate 3D models from text descriptions")
    parser.add_argument("prompt", help="Text description of the model to generate")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for generated files")
    parser.add_argument("--openscad-path", default=None, 
                      help="Path to OpenSCAD executable (overrides environment variable)")
    parser.add_argument("--model", "-m", default="llama4", 
                      help="Ollama model to use (default: llama4)")
    
    args = parser.parse_args()
    
    # Set OpenSCAD path if provided
    if args.openscad_path:
        os.environ["OPENSCAD_BINARY"] = args.openscad_path
        
    # Set Ollama model
    os.environ["OLLAMA_MODEL"] = args.model
    
    # Check what parameters our generate_model function accepts
    sig = inspect.signature(generate_model)
    valid_params = {}
    
    # Only pass parameters that the function accepts
    if 'prompt' in sig.parameters:
        valid_params['prompt'] = args.prompt
    
    if 'output_dir' in sig.parameters:
        valid_params['output_dir'] = args.output_dir
    
    # Generate the model with valid parameters
    result = generate_model(**valid_params)
    
    # Print results
    if result["errors"]:
        print("Errors:")
        for error in result["errors"]:
            print(f"  - {error}")
    
    print("\nGenerated files:")
    for file_type, file_path in result.items():
        if file_type != "errors" and file_path:
            print(f"  - {file_type}: {file_path}")
    
    return 0 if not result["errors"] else 1

if __name__ == "__main__":
    sys.exit(main())
