"""
API module for interfacing with Ollama models
"""
import requests
import os
import logging
import json
import tempfile
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Centralized configuration
class Config:
    """Central configuration for API settings and defaults"""
    # Base URLs
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Model defaults
    DEFAULT_TEXT_MODEL = os.environ.get("DEFAULT_TEXT_MODEL", "qwen3:latest")
    DEFAULT_CODE_MODEL = os.environ.get("DEFAULT_CODE_MODEL", "qwen3:latest")
    
    # OpenSCAD configuration
    OPENSCAD_BINARY = os.environ.get("OPENSCAD_BINARY", "openscad")
    
    # System prompts
    DEFAULT_SYSTEM_PROMPT = os.environ.get("OLLAMA_SYSTEM_PROMPT", "")
    
    # API endpoints
    @classmethod
    def generate_url(cls):
        return f"{cls.OLLAMA_BASE_URL}/api/generate"
    
    @classmethod
    def tags_url(cls):
        return f"{cls.OLLAMA_BASE_URL}/api/tags"
    
    # Helper methods
    @classmethod
    def get_model(cls, model=None, for_code=False):
        """Get the model to use, with appropriate fallbacks"""
        if model:
            return model
        
        # Check environment for any model override
        env_model = os.environ.get("OLLAMA_MODEL")
        if env_model:
            return env_model
            
        # Use appropriate default based on task
        return cls.DEFAULT_CODE_MODEL if for_code else cls.DEFAULT_TEXT_MODEL

def generate_text(prompt, model=None):
    """
    Generate text using Ollama API
    
    Parameters:
    - prompt (str): The prompt to send to the model
    - model (str, optional): The Ollama model to use. If None, uses environment variable or default
    
    Returns:
    - str: The generated text
    """
    # Get model from centralized config
    model = Config.get_model(model)
    
    logger.info(f"Generating text with model: {model}")
    
    try:
        url = Config.generate_url()
        
        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if available
        if Config.DEFAULT_SYSTEM_PROMPT:
            payload["system"] = Config.DEFAULT_SYSTEM_PROMPT
        
        # Make the API request
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}: {response.text}")
            return f"Error: API request failed with status code {response.status_code}"
        
        # Extract the response
        result = response.json()
        return result.get("response", "")
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return f"Error: {str(e)}"

def generate_model(prompt, model=None):
    """
    Generate a 3D model using OpenSCAD based on a text prompt
    
    Parameters:
    - prompt (str): The description of the 3D model to generate
    - model (str, optional): The Ollama model to use. If None, uses environment variable or default
    
    Returns:
    - dict: Dictionary containing paths to generated files and any errors
    """
    # Get model from centralized config, specifying it's for code generation
    model = Config.get_model(model, for_code=True)
    
    logger.info(f"Generating 3D model with: {model}")
    
    # Create a temporary directory for our files
    temp_dir = tempfile.mkdtemp()
    
    # Define output file paths
    scad_file = Path(temp_dir) / "model.scad"
    stl_file = Path(temp_dir) / "model.stl"
    png_file = Path(temp_dir) / "model.png"
    
    # Initialize result dictionary
    result = {
        "scad": str(scad_file),
        "stl": str(stl_file),
        "png": str(png_file),
        "errors": []
    }
    
    try:
        # Generate OpenSCAD code using the selected model
        logger.info(f"Generating OpenSCAD code for: {prompt}")
        
        # Prepare the request payload
        url = Config.generate_url()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if available
        if Config.DEFAULT_SYSTEM_PROMPT:
            payload["system"] = Config.DEFAULT_SYSTEM_PROMPT
        
        # Make the API request
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            error_msg = f"API request failed with status code {response.status_code}: {response.text}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result
        
        # Extract the OpenSCAD code from the response
        openscad_code = response.json().get("response", "")
        
        if not openscad_code:
            error_msg = "Received empty response from Ollama"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result
        
        # Clean up the OpenSCAD code by removing any non-code content
        # Typical for code generation models that might add commentary
        if "```" in openscad_code:
            # Extract code from Markdown code blocks
            code_blocks = openscad_code.split("```")
            # If there are multiple blocks, find the one that looks like OpenSCAD
            if len(code_blocks) > 1:
                for block in code_blocks:
                    if "module" in block or "cube" in block or "cylinder" in block or "sphere" in block:
                        openscad_code = block.strip()
                        break
        
        # Write OpenSCAD code to file
        with open(scad_file, "w") as f:
            f.write(openscad_code)
        
        logger.info(f"OpenSCAD code written to: {scad_file}")
        
        # Generate STL file using OpenSCAD
        logger.info(f"Generating STL file using OpenSCAD: {Config.OPENSCAD_BINARY}")
        stl_cmd = [Config.OPENSCAD_BINARY, "-o", str(stl_file), str(scad_file)]
        stl_result = subprocess.run(stl_cmd, capture_output=True, text=True)
        
        if stl_result.returncode != 0:
            error_msg = f"OpenSCAD STL generation failed: {stl_result.stderr}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            # Continue to try rendering the PNG even if STL generation failed
        
        # Generate PNG preview using OpenSCAD
        logger.info(f"Generating PNG preview using OpenSCAD")
        png_cmd = [Config.OPENSCAD_BINARY, "--colorscheme", "Tomorrow Night", "--camera", "0,0,0,55,0,25,140", "-o", str(png_file), str(scad_file)]
        png_result = subprocess.run(png_cmd, capture_output=True, text=True)
        
        if png_result.returncode != 0:
            error_msg = f"OpenSCAD PNG generation failed: {png_result.stderr}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
        
        return result
        
    except Exception as e:
        error_msg = f"Error during model generation: {str(e)}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
        return result

# Additional utility functions for checking available models
def get_available_ollama_models():
    """Fetch available Ollama models using the API"""
    try:
        response = requests.get(Config.tags_url())
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        else:
            logger.error(f"Failed to fetch models: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
        return []
