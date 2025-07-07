"""
Configuration settings for the Local AI 3D Model Generator
"""

import os

# Default settings
SETTINGS = {
    # Ollama model to use for code generation
    "ollama_model": "codellama:7b",
    
    # Output directory for generated files
    "output_dir": os.path.join(os.getcwd(), "output"),
    
    # OpenSCAD executable path (if not in PATH)
    "openscad_path": "",  # Leave empty to use PATH
    
    # PNG rendering settings
    "png_width": 1024,
    "png_height": 768,
}

# Override settings from environment variables if present
if os.getenv("OLLAMA_MODEL"):
    SETTINGS["ollama_model"] = os.getenv("OLLAMA_MODEL")

if os.getenv("OUTPUT_DIR"):
    SETTINGS["output_dir"] = os.getenv("OUTPUT_DIR")
