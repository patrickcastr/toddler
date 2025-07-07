"""
Helper functions for OpenSCAD integration
"""

import os
import subprocess
import platform

def find_openscad_executable():
    """
    Find the OpenSCAD executable on the system
    """
    possible_paths = []
    
    if platform.system() == "Windows":
        # Common Windows installation paths
        possible_paths = [
            "C:\\Program Files\\OpenSCAD\\openscad.com",
            "C:\\Program Files\\OpenSCAD\\openscad.exe",
            "C:\\Program Files (x86)\\OpenSCAD\\openscad.com",
            "C:\\Program Files (x86)\\OpenSCAD\\openscad.exe",
        ]
    elif platform.system() == "Darwin":  # macOS
        possible_paths = [
            "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
        ]
    else:  # Linux and others
        possible_paths = [
            "/usr/bin/openscad",
            "/usr/local/bin/openscad"
        ]
        
        # Try using which command on Unix-like systems
        try:
            which_result = subprocess.run(["which", "openscad"], 
                                         capture_output=True, 
                                         text=True, 
                                         check=False)
            if which_result.returncode == 0:
                possible_paths.insert(0, which_result.stdout.strip())
        except Exception:
            pass
    
    # Return the first path that exists
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    # If no paths exist, return the most common default
    if platform.system() == "Windows":
        return "C:\\Program Files (x86)\\OpenSCAD\\openscad.com"
    elif platform.system() == "Darwin":
        return "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
    else:
        return "/usr/bin/openscad"

# Example usage
if __name__ == "__main__":
    print(f"Found OpenSCAD at: {find_openscad_executable()}")
