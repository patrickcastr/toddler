#!/usr/bin/env python3
"""
Local 3D Model Generator using Ollama + OpenSCAD

This script takes a natural language prompt (e.g. "a simple rectangular tray with curved edges"),
uses a local LLM via Ollama to generate OpenSCAD code, and:
- Exports the design to .scad
- Uses the OpenSCAD CLI to generate .stl and .png preview

Dependencies:
- Python 3.8+
- Ollama (https://ollama.com)
- OpenSCAD CLI installed and in PATH
"""

import argparse
import json
import os
import requests
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Dict, Tuple

from config import SETTINGS
from utils import extract_openscad_code, create_openscad_prompt

def check_dependencies():
    """Verify that required dependencies are available."""
    # Check for Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
        print("ERROR: Ollama server not running. Please start Ollama first.")
        print("       Visit https://ollama.com for installation instructions.")
        sys.exit(1)
        
    # Check for OpenSCAD
    if not shutil.which("openscad"):
        print("ERROR: OpenSCAD not found in PATH")
        print("       Please install OpenSCAD from https://openscad.org/")
        sys.exit(1)
    
    print("✓ All dependencies found")

def generate_openscad_code(prompt: str) -> str:
    """
    Sends the user prompt to a local Ollama model and returns OpenSCAD code.
    Uses a specialized prompt template for better results.
    """
    full_prompt = create_openscad_prompt(prompt)
    
    try:
        print(f"🧠 Asking {SETTINGS['ollama_model']} to generate OpenSCAD code...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": SETTINGS["ollama_model"], 
                  "prompt": full_prompt,
                  "stream": False}
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract the actual OpenSCAD code from the LLM response
        code = extract_openscad_code(result['response'])
        
        if not code:
            print("WARNING: Couldn't extract proper OpenSCAD code from the response.")
            print("Using full response as code (this might not work correctly).")
            code = result['response'].strip()
        
        return code
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to communicate with Ollama: {e}")
        sys.exit(1)

def save_and_export(scad_code: str, output_dir: str, base_name: str = None) -> Dict[str, str]:
    """
    Saves the OpenSCAD code and exports STL and PNG using OpenSCAD CLI.
    Returns a dictionary with paths to all generated files.
    """
    if base_name is None:
        base_name = f"model_{uuid.uuid4().hex[:8]}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create file paths
    scad_file = os.path.join(output_dir, f"{base_name}.scad")
    stl_file = os.path.join(output_dir, f"{base_name}.stl")
    png_file = os.path.join(output_dir, f"{base_name}.png")
    
    # Write OpenSCAD file
    try:
        with open(scad_file, "w") as f:
            f.write(scad_code)
        print(f"✓ OpenSCAD file written: {scad_file}")
    except IOError as e:
        print(f"ERROR: Failed to write SCAD file: {e}")
        sys.exit(1)

    # Export to STL
    try:
        subprocess.run(
            ["openscad", "-o", stl_file, scad_file], 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✓ STL file generated: {stl_file}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: OpenSCAD failed to generate STL: {e}")
        print(f"OpenSCAD stderr: {e.stderr.decode()}")
        return {"scad": scad_file}  # Return what we have so far

    # Export PNG preview
    try:
        subprocess.run([
            "openscad", 
            "--imgsize=1024,768", 
            "--projection=perspective", 
            "--view=axes", 
            "--colorscheme=Tomorrow Night",
            "-o", png_file, 
            scad_file
        ], check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
        print(f"✓ PNG preview generated: {png_file}")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to generate PNG preview: {e}")
        print(f"OpenSCAD stderr: {e.stderr.decode()}")
    
    return {
        "scad": scad_file,
        "stl": stl_file,
        "png": png_file
    }

def generate_model(prompt: str, output_dir: str = "./output", name: str = None) -> Dict[str, str]:
    """
    Main function to process the prompt and generate STL and PNG files.
    Returns a dictionary with paths to all generated files.
    """
    print(f"🔍 Generating 3D model from prompt: \"{prompt}\"")
    
    # Sanitize input for filename if no custom name provided
    if name is None:
        # Create a filename-safe version of the prompt (first few words)
        words = prompt.split()[:3]
        name = "_".join("".join(c for c in word if c.isalnum()) for word in words)
        name = f"{name}_{uuid.uuid4().hex[:6]}"
    
    # Generate OpenSCAD code
    scad_code = generate_openscad_code(prompt)
    
    # Save and export the model
    return save_and_export(scad_code, output_dir, name)

def main():
    parser = argparse.ArgumentParser(description="Generate 3D models from natural language prompts using local AI")
    parser.add_argument("prompt", nargs="?", help="Description of the 3D model to generate")
    parser.add_argument("--output", "-o", default=SETTINGS["output_dir"], help="Output directory for generated files")
    parser.add_argument("--name", "-n", help="Base name for output files (without extension)")
    parser.add_argument("--model", "-m", help=f"Ollama model to use (default: {SETTINGS['ollama_model']})")
    
    args = parser.parse_args()
    
    # Override model if specified
    if args.model:
        SETTINGS["ollama_model"] = args.model
    
    # Check dependencies first
    check_dependencies()
    
    # If no prompt provided, enter interactive mode
    if not args.prompt:
        print("🤖 Local AI 3D Model Generator")
        print(f"Using model: {SETTINGS['ollama_model']}")
        print("Type 'exit' to quit")
        
        while True:
            try:
                prompt = input("\n🔮 Describe your 3D model: ")
                if prompt.lower() in ["exit", "quit", "q"]:
                    break
                if not prompt.strip():
                    continue
                    
                start_time = time.time()
                files = generate_model(prompt, args.output, args.name)
                elapsed = time.time() - start_time
                
                print(f"✨ Done in {elapsed:.1f} seconds!")
                print(f"📁 Files generated:")
                for file_type, file_path in files.items():
                    print(f"  • {file_type.upper()}: {file_path}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"ERROR: {str(e)}")
    else:
        # Process a single prompt
        files = generate_model(args.prompt, args.output, args.name)
        print(f"📁 Files generated:")
        for file_type, file_path in files.items():
            print(f"  • {file_type.upper()}: {file_path}")

if __name__ == "__main__":
    main()
