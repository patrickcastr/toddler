"""
Utility functions for the Local AI 3D Model Generator
"""

import re

def extract_openscad_code(llm_response: str) -> str:
    """
    Extracts OpenSCAD code from the LLM response.
    Handles different code block formats that the LLM might use.
    """
    # Try to find code between ```openscad and ``` markers
    pattern1 = r'```(?:openscad|scad)(.*?)```'
    match = re.search(pattern1, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find any code between ``` markers
    pattern2 = r'```(.*?)```'
    match = re.search(pattern2, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Check if the response likely contains only code (has typical OpenSCAD functions/syntax)
    if any(keyword in llm_response for keyword in ['module ', 'function ', 'cube(', 'cylinder(', 'sphere(']):
        return llm_response.strip()
    
    return ""

def create_openscad_prompt(user_prompt: str) -> str:
    """
    Creates a specialized prompt for the LLM to generate OpenSCAD code.
    This provides clear instructions to the model about what kind of code to generate.
    """
    return f"""Generate OpenSCAD code for a 3D model of: "{user_prompt}"

You are an expert in parametric 3D modeling with OpenSCAD. Create a complete and working OpenSCAD script that can be directly executed.

Requirements:
1. Use only valid OpenSCAD syntax
2. Include proper module definitions and parameters where appropriate
3. Use variables for important dimensions to make the model customizable
4. Add brief comments explaining key components
5. Focus on clean, efficient code
6. Include only the code, not explanations
7. The model should be positioned at the origin (0,0,0)
8. The model should be properly oriented for 3D printing (flat bottom if applicable)

Only respond with the OpenSCAD code. Do not include markdown formatting, explanations, or backticks.
"""
