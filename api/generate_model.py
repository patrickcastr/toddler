#!/usr/bin/env python3
"""
Local 3D Model Generator using Ollama + OpenSCAD

This script takes a natural language prompt and generates 3D models using local LLM.
"""

import json
import os
import requests
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path

from .config import SETTINGS
from .utils import extract_openscad_code, create_openscad_prompt

# Import our OpenSCAD utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from openscad_util import render_openscad

def check_dependencies() -> Dict[str, bool]:
    """
    Verify that required dependencies are available.
    Returns dict with status of each dependency.
    """
    dependencies = {
        "ollama": False,
        "openscad": False
    }
    
    # Check for Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        dependencies["ollama"] = True
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError, requests.exceptions.Timeout):
        pass
        
    # Check for OpenSCAD
    if shutil.which("openscad"):
        dependencies["openscad"] = True
    
    return dependencies

def generate_openscad_code(prompt: str) -> Tuple[str, Optional[str]]:
    """
    Sends the user prompt to a local Ollama model and returns OpenSCAD code.
    Returns tuple of (code, error_message)
    """
    full_prompt = create_openscad_prompt(prompt)
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": SETTINGS["ollama_model"], 
                  "prompt": full_prompt,
                  "stream": False},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract the actual OpenSCAD code from the LLM response
        code = extract_openscad_code(result['response'])
        
        if not code:
            return result['response'].strip(), "Warning: Couldn't extract proper OpenSCAD code"
        
        return code, None
        
    except requests.exceptions.RequestException as e:
        return "", f"Failed to communicate with Ollama: {str(e)}"

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
    
    result = {"scad": "", "stl": "", "png": "", "errors": []}
    
    # Write OpenSCAD file
    try:
        with open(scad_file, "w") as f:
            f.write(scad_code)
        result["scad"] = scad_file
    except IOError as e:
        result["errors"].append(f"Failed to write SCAD file: {str(e)}")
        return result

    # Use the render_openscad utility instead of direct subprocess calls
    render_result = render_openscad(
        scad_code=scad_code,
        output_stl=stl_file,
        output_png=png_file,
        openscad_path=os.environ.get("OPENSCAD_BINARY")
    )
    
    # Check for errors in rendering
    if render_result["errors"]:
        result["errors"].extend(render_result["errors"])
    
    result["stl"] = render_result["stl"]
    result["png"] = render_result["png"]
    
    return result

def generate_model(prompt: str, output_dir: str = None, name: str = None) -> Dict[str, str]:
    """
    Main function to process the prompt and generate STL and PNG files.
    Returns a dictionary with paths to all generated files and any errors.
    """
    if output_dir is None:
        output_dir = SETTINGS["output_dir"]
    
    # Sanitize input for filename if no custom name provided
    if name is None:
        # Create a filename-safe version of the prompt (first few words)
        words = prompt.split()[:3]
        name = "_".join("".join(c for c in word if c.isalnum()) for word in words)
        name = f"{name}_{uuid.uuid4().hex[:6]}"
    
    # Generate OpenSCAD code
    scad_code, error = generate_openscad_code(prompt)
    
    result = {"scad": "", "stl": "", "png": "", "errors": []}
    if error:
        result["errors"].append(error)
    
    # If we got some code, try to save and export it
    if scad_code:
        export_result = save_and_export(scad_code, output_dir, name)
        
        # Merge the results
        result["scad"] = export_result.get("scad", "")
        result["stl"] = export_result.get("stl", "")
        result["png"] = export_result.get("png", "")
        
        if "errors" in export_result:
            result["errors"].extend(export_result["errors"])
    
    return result

if __name__ == "__main__":
    # CLI handling code remains for standalone usage
    import argparse
    
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
    dependencies = check_dependencies()
    if not all(dependencies.values()):
        missing = [dep for dep, status in dependencies.items() if not status]
        print(f"ERROR: Missing dependencies: {', '.join(missing)}")
        sys.exit(1)
        
    # Call the function with CLI arguments
    if args.prompt:
        result = generate_model(args.prompt, args.output, args.name)
        
        if result["errors"]:
            print("Errors occurred:")
            for error in result["errors"]:
                print(f"- {error}")
                
        print(f"Files generated:")
        for file_type in ["scad", "stl", "png"]:
            if result[file_type]:
                print(f"  • {file_type.upper()}: {result[file_type]}")
    else:
        parser.print_help()
