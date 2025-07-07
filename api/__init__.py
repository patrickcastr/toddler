"""
API module for Local AI 3D Model Generator
"""

from .generate_model import generate_model, check_dependencies
import os
import requests
import logging

__all__ = ["generate_model", "check_dependencies"]

# Configure logging if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_text(prompt, temperature=0.7, max_tokens=4000, gpu_options=None):
    """
    Generate text using the Ollama API based on the provided prompt.
    
    Args:
        prompt: The text prompt to process
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum tokens to generate in response
        gpu_options: Dictionary with GPU settings (num_gpu, gpu_layers, f16)
        
    Returns:
        The generated text as a string, or a dict with an error message on failure
    """
    # Get model from environment or use default
    model = os.environ.get("OLLAMA_MODEL", "llama4:latest")
    system_prompt = os.environ.get("OLLAMA_SYSTEM_PROMPT", """
    You are an expert document processor and text organizer.
    Your task is to take extracted text snippets from documents and convert them into:
    1. Well-structured paragraphs
    2. Coherent summaries
    3. Meaningful insights
    
    Maintain all the factual information from the original text.
    Format your response neatly with proper headings and sections where appropriate.
    """)
    
    logger.info(f"Generating text using model: {model}")
    
    # Default GPU options
    default_gpu_options = {
        "num_gpu": 1,        # Number of GPUs to use
        "num_batch": 128,    # Batch size for processing
    }
    
    # Update with user-provided options if any
    if gpu_options:
        default_gpu_options.update(gpu_options)
        
    # Log GPU options being used
    logger.info(f"Using GPU options: {default_gpu_options}")
    
    # Prepare API request with GPU configuration
    api_url = "http://localhost:11434/api/generate"
    request_data = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "temperature": temperature,
        "num_predict": max_tokens,
        "stream": False,
        "options": default_gpu_options
    }
    
    try:
        # Make API request
        logger.info(f"Sending request to Ollama API for processing large text")
        response = requests.post(api_url, json=request_data)
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {"error": error_msg}
            
        # Parse the response
        result = response.json()
        
        if "response" in result:
            logger.info("Successfully generated text")
            return result["response"]
        else:
            error_msg = "No response field in Ollama API result"
            logger.error(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Error generating text: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
