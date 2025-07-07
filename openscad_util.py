"""
Utilities for working with OpenSCAD
"""
import os
import subprocess
import platform
import tempfile
import logging

logger = logging.getLogger("3dgen.openscad")

def render_openscad(scad_code, output_stl=None, output_png=None, openscad_path=None):
    """
    Renders OpenSCAD code to STL and/or PNG files
    
    Args:
        scad_code (str): OpenSCAD code to render
        output_stl (str, optional): Path to output STL file
        output_png (str, optional): Path to output PNG file
        openscad_path (str, optional): Path to OpenSCAD executable
        
    Returns:
        dict: Paths to generated files
    """
    # Get OpenSCAD path
    if not openscad_path:
        openscad_path = os.environ.get("OPENSCAD_BINARY")
        
    if not openscad_path:
        # Try to find OpenSCAD
        if platform.system() == "Windows":
            for path in [
                "C:\\Program Files\\OpenSCAD\\openscad.com",
                "C:\\Program Files (x86)\\OpenSCAD\\openscad.com",
                "C:\\Program Files\\OpenSCAD\\openscad.exe",
                "C:\\Program Files (x86)\\OpenSCAD\\openscad.exe",
            ]:
                if os.path.exists(path):
                    openscad_path = path
                    break
        elif platform.system() == "Darwin":  # macOS
            openscad_path = "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
        else:  # Linux
            openscad_path = "/usr/bin/openscad"
    
    if not os.path.exists(openscad_path):
        raise FileNotFoundError(f"OpenSCAD executable not found at {openscad_path}")
    
    # Write SCAD code to temp file
    with tempfile.NamedTemporaryFile(suffix=".scad", delete=False, mode="w") as f:
        f.write(scad_code)
        scad_file = f.name
    
    results = {"scad": scad_file, "stl": None, "png": None, "errors": []}
    
    try:
        # Render STL if requested
        if output_stl:
            cmd = [openscad_path, "-o", output_stl, scad_file]
            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if process.returncode != 0:
                results["errors"].append(f"STL generation error: {process.stderr}")
            else:
                results["stl"] = output_stl
                
        # Render PNG if requested
        if output_png:
            cmd = [openscad_path, "--render", "-o", output_png, scad_file]
            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if process.returncode != 0:
                results["errors"].append(f"PNG generation error: {process.stderr}")
            else:
                results["png"] = output_png
                
        return results
    except Exception as e:
        results["errors"].append(f"OpenSCAD rendering error: {str(e)}")
        return results
