"""
Streamlit UI for Local AI 3D Model Generator
"""

import os
import sys
import base64
import streamlit as st
import time
import logging
import io
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Add the parent directory to the path so we can import from api and tools
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import generate_model at the module level to avoid scope issues
from api import generate_model

# Set page config
st.set_page_config(
    page_title="ToddLer Energy Assistant by Patrick",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_file_download_link(file_path, link_text):
    """Generate a link to download a file"""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    file_name = os.path.basename(file_path)
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'

# Set up logging
def setup_logger():
    """Set up a logger that captures logs to a string buffer"""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger("3dgen")
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.addHandler(handler)
    
    return logger, log_stream

def enhance_prompt(prompt, system_prompt=None):
    """
    Process user prompt to better communicate requirements to the model
    """
    # Use provided system prompt or create default
    enhanced_system_prompt = system_prompt or """
    Generate OpenSCAD code for a 3D model with the following specifications. Pay special attention to:

    1. COLORS: Use the color() module with appropriate RGB values or named colors
    2. HOLLOW OBJECTS: Create hollow objects by differencing a solid from a smaller version of itself
    3. DIMENSIONS: Apply all dimensions exactly as specified
    4. PARAMETERS: Place all parameters at the top as variables for easy modification

    Start your code with clear parameter definitions and include detailed comments.
    Use proper OpenSCAD modules for organization.
    """
    
    # Extract key features from the prompt to emphasize
    color_keywords = ["red", "green", "blue", "yellow", "black", "white", "orange", "purple"]
    structure_keywords = ["hollow", "solid", "empty", "filled", "shell"]
    # Add circle keywords
    shape_keywords = {
        "circle": "cylinder with small height (h=1) and appropriate radius/diameter",
        "round": "cylinder or sphere depending on context",
        "cylinder": "cylinder with appropriate height and radius/diameter",
        "sphere": "sphere with appropriate radius/diameter",
        "disc": "cylinder with small height (h=1) and appropriate radius/diameter"
    }
    
    # Identify features in the prompt
    found_colors = [color for color in color_keywords if color in prompt.lower()]
    found_structures = [struct for struct in structure_keywords if struct in prompt.lower()]
    found_shapes = {shape: desc for shape, desc in shape_keywords.items() if shape in prompt.lower()}
    
    # Create a structured prompt that emphasizes the important features
    structured_prompt = f"""
    Create OpenSCAD code for the following 3D model:
    
    {prompt}
    
    IMPORTANT: Create ONLY what is described above. Follow the exact dimensions and features requested.
    """
    
    if found_colors:
        structured_prompt += f"\n- COLOR: The model should be {', '.join(found_colors)}"
    
    if found_structures:
        structured_prompt += f"\n- STRUCTURE: The model should be {', '.join(found_structures)}"
        if "hollow" in found_structures or "empty" in found_structures:
            structured_prompt += " (create this using difference() between outer and inner shapes)"
    
    if found_shapes:
        structured_prompt += "\n- SHAPES DETECTED:"
        for shape, desc in found_shapes.items():
            structured_prompt += f"\n  - {shape.upper()}: Create using {desc}"
    
    # Add examples in a way that makes it clear they are for format reference only
    structured_prompt += """
    
    Your code should follow this structure pattern but ONLY implement what was specifically requested:
    
    // Parameters section - define dimensions
    <parameter_name> = <value>; // units
    
    // Main module section - implement the requested shape
    module <descriptive_name>() {
        // Use appropriate OpenSCAD functions based on the request
    }
    
    // Call the module
    <descriptive_name>();
    
    For example, if someone requested a basic cube, you might write:
    
    // Parameters
    size = 20; // mm
    
    // Main module
    module basic_cube() {
        cube([size, size, size], center=true);
    }
    
    // Create the shape
    basic_cube();
    
    For a circle with 20mm diameter, you would write:
    
    // Parameters
    diameter = 20; // mm
    height = 1; // mm - keep small for a flat circle
    
    // Main module
    module flat_circle() {
        cylinder(h=height, d=diameter, center=true, $fn=100);
    }
    
    // Create the shape
    flat_circle();
    
    REMEMBER: Only implement EXACTLY what was requested in the original description, no more and no less.
    Only provide valid OpenSCAD code without explanations or remarks.
    Include helpful comments within the code.
    """
    
    return structured_prompt, enhanced_system_prompt

def show_main_menu():
    """Display the main menu with navigation buttons"""
    st.title("🧊 ToddLer Energy Assistant - Patrick")
    
    st.markdown("""
    ## Welcome to the ToddLer Energy Assistant!
    This app allows you to analyze energy projects, process documents, and generate 3D models using AI.
    Choose one of the available tools below:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### Economic Model")
        st.markdown("Build financial models for energy projects")
        if st.button("💰 Launch Economic Model", use_container_width=True):
            st.session_state.current_page = "economic_model"
            st.rerun()
    
    with col2:
        st.info("### Document Processing")
        st.markdown("Extract and process text from documents using AI")
        if st.button("📄 Launch Document Processor", use_container_width=True):
            st.session_state.current_page = "assistant"
            st.rerun()
            
    with col3:
        st.info("### 3D Model Generator")
        st.markdown("Generate 3D models from text descriptions using OpenSCAD")
        if st.button("🧊 Launch 3D Model Generator", use_container_width=True):
            st.session_state.current_page = "model_generator"
            st.rerun()

def get_available_ollama_models():
    """Fetch available Ollama models using the API"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        else:
            return []
    except Exception as e:
        print(f"Error fetching Ollama models: {str(e)}")
        return []

def show_model_generator():
    """Show the 3D model generator interface"""
    st.title("🧊 Local AI 3D Model Generator - Patrick")
    
    # Add navigation back to main menu
    if st.button("← Back to Main Menu"):
        st.session_state.current_page = "main_menu"
        st.rerun()
    
    # Set up logger
    logger, log_stream = setup_logger()
    
    # Initialize log storage in session state
    if "logs" not in st.session_state:
        st.session_state.logs = []
    
    # Add sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Initialize session state for openscad path if not exists
        if "openscad_path" not in st.session_state:
            # Try to find the most likely OpenSCAD path
            default_path = "C:\\Program Files (x86)\\OpenSCAD\\openscad.exe"
            com_path = "C:\\Program Files (x86)\\OpenSCAD\\openscad.com"
            
            if os.path.exists(com_path):
                default_path = com_path
            
            st.session_state["openscad_path"] = default_path
            
        openscad_path = st.text_input(
            "OpenSCAD executable path", 
            value=st.session_state["openscad_path"],
            help="Path to the OpenSCAD executable file (.exe or .com)"
        )
        
        # Update session state when changed
        if openscad_path != st.session_state["openscad_path"]:
            st.session_state["openscad_path"] = openscad_path
            
            # Try to automatically fix common issues
            if openscad_path.endswith(".exe") and not os.path.exists(openscad_path):
                com_path = openscad_path.replace(".exe", ".com")
                if os.path.exists(com_path):
                    st.session_state["openscad_path"] = com_path
                    openscad_path = com_path
                    st.info("Automatically switched to .com version which was found.")
            elif openscad_path.endswith(".com") and not os.path.exists(openscad_path):
                exe_path = openscad_path.replace(".com", ".exe")
                if os.path.exists(exe_path):
                    st.session_state["openscad_path"] = exe_path
                    openscad_path = exe_path
                    st.info("Automatically switched to .exe version which was found.")
        
        # Add a visual confirmation of OpenSCAD status
        if os.path.exists(openscad_path):
            st.success(f"✅ OpenSCAD found at: {openscad_path}")
            
            # Show additional help if using .exe instead of .com
            if openscad_path.endswith(".exe"):
                com_path = openscad_path.replace(".exe", ".com")
                if os.path.exists(com_path):
                    st.info(f"Note: There's also a .com version at {com_path} which may be needed.")
        else:
            st.warning("⚠️ OpenSCAD not found at the specified path")
            st.info("""
            Try searching for both:
            - openscad.exe (graphical version)
            - openscad.com (command-line version)
            
            The system might be looking for the .com version specifically.
            """)
    
        # Add Ollama configuration section
        st.subheader("Ollama")
        st.markdown("""
        This app requires [Ollama](https://ollama.com/download) with the codellama:7b model.
        
        If you're seeing connection errors:
        1. Make sure Ollama is running
        2. Open a terminal and run: `ollama pull codellama:7b`
        """)
        
        # Show advanced options in an expander
        with st.expander("Advanced Model Settings"):
            # Add model selection with dropdown instead of text input
            if "selected_model" not in st.session_state:
                st.session_state.selected_model = "codellama:7b"
                
            # Get available models
            if "available_models" not in st.session_state:
                st.session_state.available_models = get_available_ollama_models()
            
            # Add refresh button for model list
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Select Ollama Model:**")
            with col2:
                if st.button("🔄 Refresh"):
                    st.session_state.available_models = get_available_ollama_models()
                    st.rerun()
            
            # Show dropdown with available models if any are found
            if st.session_state.available_models:
                selected_model = st.selectbox(
                    "Available Models",
                    options=st.session_state.available_models,
                    index=st.session_state.available_models.index(st.session_state.selected_model) 
                        if st.session_state.selected_model in st.session_state.available_models 
                        else 0,
                    help="Select an Ollama model to use"
                )
                
                if selected_model != st.session_state.selected_model:
                    st.session_state.selected_model = selected_model
            else:
                # Fallback to text input if no models are found
                st.warning("⚠️ No Ollama models found. Please check your Ollama installation.")
                selected_model = st.text_input(
                    "Ollama Model",
                    value=st.session_state.selected_model,
                    help="Name of the Ollama model to use (e.g., llama4, codellama, mistral)"
                )
                
                if selected_model != st.session_state.selected_model:
                    st.session_state.selected_model = selected_model
                
                st.info("After installing new models, click 'Refresh' to update the list.")
            
            # Option to customize system prompt
            st.subheader("System Prompt")
            default_system_prompt = """You are an expert OpenSCAD developer who creates flawless 3D models.
Follow these precise requirements:

1. Use primitive shapes correctly:
   - cube([x,y,z]) for rectangular objects
   - cylinder(h=height, r=radius) or cylinder(h=height, d=diameter) for circular objects
   - sphere(r=radius) or sphere(d=diameter) for spherical objects
2. To create a CIRCLE, use a cylinder with small height: cylinder(h=1, r=radius, $fn=100)
3. Set parameters at the top with descriptive names and correct units
4. For colors, use either named colors (e.g., "red") or RGB values (e.g., [1,0,0])
5. For hollow objects, ALWAYS create an outer shape and subtract an inner shape using difference()
6. Use proper OpenSCAD operations: union(), difference(), intersection()
7. Include correct transformations: translate(), rotate(), scale()
8. Use modules to organize complex code
9. Ensure all dimensions are accurately specified
10. Always set appropriate $fn values (64-128) for smooth circles and cylinders

YOUR OUTPUT MUST BE ONLY VALID OPENSCAD CODE WITH HELPFUL COMMENTS.
"""
            
            if "system_prompt" not in st.session_state:
                st.session_state.system_prompt = default_system_prompt
                
            custom_system_prompt = st.text_area(
                "Custom System Prompt",
                value=st.session_state.system_prompt,
                height=200,
                help="Customize the system prompt sent to the model to guide the generation"
            )
            
            if custom_system_prompt != st.session_state.system_prompt:
                st.session_state.system_prompt = custom_system_prompt
        
        # Add Ollama connection check button in Configuration section
        st.subheader("Ollama Connection")
        if st.button("Check Ollama Connection"):
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    available_models = [model["name"] for model in response.json()["models"]]
                    current_model = st.session_state.get("selected_model", "codellama:7b")
                    if current_model in available_models:
                        st.success(f"✅ Connected to Ollama and {current_model} model is available")
                    else:
                        st.warning(f"⚠️ Connected to Ollama but {current_model} model is not available. Available models: {', '.join(available_models)}")
                else:
                    st.error(f"❌ Failed to fetch models from Ollama: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Failed to connect to Ollama: {str(e)}")
                st.info("Make sure Ollama is running. Download from https://ollama.com/download")
    
    # User input
    prompt = st.text_area("Describe the 3D model you want to create:", 
                         placeholder="e.g., a cylindrical cup with a handle",
                         height=100)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        generate_button = st.button("🚀 Generate Model", type="primary", disabled=not prompt)
    
    # Store results in session state
    if "results" not in st.session_state:
        st.session_state.results = None
    
    # Process generation
    if generate_button and prompt:
        with st.spinner("Generating your 3D model... This might take a minute."):
            try:
                # Record start time
                start_time = time.time()
                
                # Enhance the prompt using the custom system prompt
                enhanced_prompt, system_prompt = enhance_prompt(prompt, st.session_state.system_prompt)
                logger.info(f"Original prompt: {prompt}")
                logger.info(f"Enhanced prompt: {enhanced_prompt}")
                
                logger.info(f"Using OpenSCAD path: {os.environ.get('OPENSCAD_BINARY', 'Not set in environment')}")
                
                # Try to set environment variable for OpenSCAD path
                os.environ["OPENSCAD_BINARY"] = openscad_path
                logger.info(f"Set OPENSCAD_BINARY environment variable to: {openscad_path}")
                
                # Set the Ollama model
                current_model = st.session_state.selected_model
                os.environ["OLLAMA_MODEL"] = current_model
                logger.info(f"Using Ollama model: {current_model}")
                
                # Try to set the system prompt for the Ollama model
                os.environ["OLLAMA_SYSTEM_PROMPT"] = system_prompt
                logger.info(f"Using system prompt: {system_prompt[:100]}...")
                
                # Capture stdout/stderr
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    # Generate the model with the enhanced prompt
                    results = generate_model(enhanced_prompt)
                
                stdout_output = stdout_buffer.getvalue()
                stderr_output = stderr_buffer.getvalue()
                
                if stdout_output:
                    logger.info(f"Standard output: {stdout_output}")
                if stderr_output:
                    logger.error(f"Standard error: {stderr_output}")
                
                # Calculate duration
                duration = time.time() - start_time
                logger.info(f"Model generation completed in {duration:.2f} seconds")
                
                # Store results in session state
                st.session_state.results = results
                
                # Store logs
                st.session_state.logs.append(log_stream.getvalue())
                
            except Exception as e:
                logger.error(f"Error during model generation: {str(e)}")
                logger.error(traceback.format_exc())
                
                # More specific error handling
                if "WinError 2" in str(e) and "system cannot find the file specified" in str(e):
                    logger.error(f"Could not find the executable at: {openscad_path}")
                    st.error("""
                    The system couldn't find OpenSCAD at the specified path. Please:
                    1. Make sure OpenSCAD is installed
                    2. Check that the path in the sidebar configuration is correct
                    3. Try restarting the application
                    
                    If you haven't installed OpenSCAD, download it from: https://openscad.org/downloads.html
                    """)
                elif "404" in str(e) and "localhost:11434" in str(e):
                    logger.error("Failed to communicate with Ollama")
                    current_model = st.session_state.selected_model
                    st.error(f"""
                    Failed to communicate with Ollama. Make sure:
                    1. Ollama is installed and running
                    2. The {current_model} model is pulled
                    
                    Run these commands in your terminal:
                    ```
                    # Start Ollama if it's not running
                    # Then pull the model
                    ollama pull {current_model}
                    ```
                    """)
                else:
                    st.error(f"Error during model generation: {str(e)}")
                
                # Try to check if OpenSCAD is working
                try:
                    import subprocess
                    logger.info(f"Attempting to check OpenSCAD version at: {openscad_path}")
                    result = subprocess.run([openscad_path, "--version"], 
                                           capture_output=True, 
                                           text=True, 
                                           check=False)
                    logger.info(f"OpenSCAD check result (return code {result.returncode})")
                    logger.info(f"OpenSCAD stdout: {result.stdout}")
                    logger.info(f"OpenSCAD stderr: {result.stderr}")
                except Exception as e2:
                    logger.error(f"Failed to check OpenSCAD: {str(e2)}")
                
                # Store logs
                st.session_state.logs.append(log_stream.getvalue())
                return
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        
        # Show errors if any
        if results["errors"]:
            st.error("Errors occurred during generation:")
            for error in results["errors"]:
                st.error(error)
        
        # If we have files, show them
        if results["png"] and os.path.exists(results["png"]):
            st.success(f"Model generated successfully!")
            
            # Display the preview image
            st.subheader("Preview")
            st.image(results["png"], use_container_width=True)
            
            # Display download links
            st.subheader("Download Files")
            col1, col2 = st.columns(2)
            
            with col1:
                if results["stl"]:
                    stl_link = get_file_download_link(results["stl"], "📥 Download STL for 3D Printing")
                    st.markdown(stl_link, unsafe_allow_html=True)
                    
            with col2:
                if results["scad"]:
                    scad_link = get_file_download_link(results["scad"], "📥 Download OpenSCAD Source")
                    st.markdown(scad_link, unsafe_allow_html=True)
            
            # Show code
            if results["scad"]:
                with st.expander("View OpenSCAD Code"):
                    with open(results["scad"], 'r') as f:
                        st.code(f.read(), language="clike")

    # Add a debug section at the bottom
    with st.expander("Debug Logs", expanded=False):
        st.subheader("Debug Information")
        st.info("This section shows detailed logs to help troubleshoot issues.")
        
        # Display OpenSCAD verification
        st.subheader("OpenSCAD Verification")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test OpenSCAD Installation"):
                try:
                    import subprocess
                    with st.spinner("Testing OpenSCAD..."):
                        result = subprocess.run([openscad_path, "--version"], 
                                               capture_output=True, 
                                               text=True, 
                                               check=False)
                        
                        if result.returncode == 0:
                            st.success(f"✅ OpenSCAD is working! Version: {result.stdout.strip()}")
                        else:
                            st.error(f"❌ OpenSCAD test failed with code {result.returncode}")
                            st.code(f"stdout: {result.stdout}\nstderr: {result.stderr}")
                except Exception as e:
                    st.error(f"Failed to execute OpenSCAD: {str(e)}")
                    st.info("""
                    Make sure:
                    1. The path is correct
                    2. You're using the right executable (.com for command line, .exe for GUI)
                    3. OpenSCAD is properly installed
                    """)
        
        # Display API information - fixed to avoid scope issues
        import inspect
        try:
            # Use the already imported generate_model instead of reimporting
            st.subheader("API Information")
            st.write(f"generate_model function signature: {str(inspect.signature(generate_model))}")
            st.write(f"generate_model docstring: {generate_model.__doc__ or 'No docstring available'}")
            # Show the module where generate_model is defined
            st.write(f"generate_model is defined in: {generate_model.__module__}")
        except Exception as e:
            st.error(f"Could not inspect API: {str(e)}")
        
        # Display environment variables
        st.subheader("Environment Variables")
        st.write({k:v for k,v in os.environ.items() if k.lower().startswith(('openscad', 'path'))})
        
        # Display logs
        if st.session_state.logs:
            st.subheader("Execution Logs")
            
            # Use tabs instead of nested expanders
            if len(st.session_state.logs) > 0:
                # Create tab labels
                log_tabs = [f"Log {i+1}" for i in range(len(st.session_state.logs))]
                
                # Create tabs for logs
                tab_selected = st.radio("Select Log", log_tabs, index=len(log_tabs)-1)
                
                # Get the index of the selected tab
                selected_index = int(tab_selected.split()[1]) - 1
                
                # Show the selected log
                st.code(st.session_state.logs[selected_index])
                
                # Add a checkbox to show/hide all logs
                if st.checkbox("Show all logs"):
                    for i, log in enumerate(st.session_state.logs):
                        st.markdown(f"**Log {i+1}**")
                        st.code(log)
        
        # Information about OpenSCAD
        st.subheader("OpenSCAD Information")
        st.markdown("""
        ### How OpenSCAD is Used
        
        OpenSCAD doesn't need to be running before you generate a model. The script will:
        1. Generate OpenSCAD code
        2. Save it to a temporary file
        3. Call OpenSCAD to render the model (to both PNG and STL)
        4. Close OpenSCAD automatically
        
        ### Common Issues
        
        - **Installation Problems**: Ensure OpenSCAD is installed and the path is correctly set in the app.
        - **Path Issues**: The app needs the correct path to the OpenSCAD executable (.exe or .com).
        - **Model Generation Errors**: Check the logs for specific error messages. Common issues include:
          - Errors in the generated code
          - Unsupported features in OpenSCAD
          - Invalid parameters in the prompt

        ### Troubleshooting
        
        If you're having issues:
        1. Make sure the path points to the right executable:
           - Use `openscad.com` for command-line operations (preferred)
           - Use `openscad.exe` only if .com doesn't work
        2. Make sure OpenSCAD can be run from the command line
        3. Try running OpenSCAD manually to verify it works
        
        For detailed troubleshooting, refer to the [OpenSCAD documentation](https://openscad.org/documentation.html) and [FAQ](https://openscad.org/faq.html).
        """)
        
        # Add button to clear logs
        if st.button("Clear Logs"):
            st.session_state.logs = []
            st.rerun()

def show_economic_model():
    """Show the economic model interface"""
    # Instead of importing the module directly, use the function to show the economic model
    def show_economic_model():
        from tools.economic_model import show_economic_model
        show_economic_model()
    
    # Show the economic model UI
    show_economic_model()

def show_assistant():
    """Document processing assistant"""
    # Dynamically import to avoid import errors
    import sys
    from pathlib import Path
    
    # Add the parent directory to the path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Import the document processor function
    from tools.document_processor import show_document_processor
    
    # Show the document processor UI
    show_document_processor()

def main():
    """Main entry point for the application"""
    # Initialize the current page in session state if it doesn't exist
    if "current_page" not in st.session_state:
        st.session_state.current_page = "main_menu"
    
    # Route to the appropriate page based on the current state
    if st.session_state.current_page == "main_menu":
        show_main_menu()
    elif st.session_state.current_page == "model_generator":
        show_model_generator()
    elif st.session_state.current_page == "assistant":
        show_assistant()
    elif st.session_state.current_page == "economic_model":
        show_economic_model()
    else:
        st.error(f"Unknown page: {st.session_state.current_page}")
        st.session_state.current_page = "main_menu"
        st.rerun()

if __name__ == "__main__":
    main()