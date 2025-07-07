"""
Document Processing Assistant
- Extracts text from documents
- Processes text snippets into coherent paragraphs
- Stores all data locally
"""

import os
import sys
import base64
import streamlit as st
import pandas as pd
import tempfile
import json
from pathlib import Path
from datetime import datetime
import uuid
import logging
import subprocess
import re
import platform
import numpy as np

# Add document processing libraries
import PyPDF2
try:
    import docx
except ImportError:
    docx = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

# Add parent directory to path for importing ollama utilities
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from api import generate_text  # Assume this exists or you'll need to create it

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc_processor")

# Data storage paths
DATA_DIR = parent_dir / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EXTRACTIONS_DIR = DATA_DIR / "extractions"
BATCH_DIR = DATA_DIR / "batches"

# Ensure directories exist
for directory in [DATA_DIR, DOCUMENTS_DIR, EXTRACTIONS_DIR, BATCH_DIR]:
    directory.mkdir(exist_ok=True)

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from a Word document"""
    if docx is None:
        return "python-docx library not installed. Install with: pip install python-docx"
    
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
    return text

def extract_text_from_txt(file_path):
    """Extract text from a plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Error reading text file: {str(e)}")
        return ""

def extract_text_from_html(file_path):
    """Extract text from an HTML file"""
    if BeautifulSoup is None:
        return "BeautifulSoup library not installed. Install with: pip install beautifulsoup4"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            # Get text
            text = soup.get_text()
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        return ""

def extract_text_from_csv(file_path):
    """Extract data from a CSV file as DataFrame and string preview."""
    try:
        df = pd.read_csv(file_path)
        return df, df.head().to_string()
    except Exception as e:
        logger.error(f"Error reading CSV: {str(e)}")
        return None, ""

def extract_text_from_excel(file_path):
    """Extract data from an Excel file as DataFrame and string preview."""
    try:
        df = pd.read_excel(file_path)
        return df, df.head().to_string()
    except Exception as e:
        logger.error(f"Error reading Excel: {str(e)}")
        return None, ""

def extract_text(file_path):
    """Extract text or data from a file based on its extension"""
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == '.docx':
        return extract_text_from_docx(file_path)
    elif file_ext == '.txt':
        return extract_text_from_txt(file_path)
    elif file_ext in ['.html', '.htm']:
        return extract_text_from_html(file_path)
    elif file_ext == '.csv':
        return extract_text_from_csv(file_path)
    elif file_ext in ['.xls', '.xlsx']:
        return extract_text_from_excel(file_path)
    else:
        return f"Unsupported file format: {file_ext}"

def chunk_text(text, max_chunk_size=4000):
    """Split text into manageable chunks for processing"""
    chunks = []
    # Try to split at paragraph boundaries
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            # If the current chunk has content, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
            
            # If paragraph itself is too long, split it further
            if len(para) > max_chunk_size:
                # Split at sentence boundaries
                sentences = para.replace('. ', '.|').split('|')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If sentence is still too long, split it into fixed-size chunks
                        if len(sentence) > max_chunk_size:
                            for i in range(0, len(sentence), max_chunk_size):
                                chunks.append(sentence[i:i+max_chunk_size])
                            current_chunk = ""
                        else:
                            current_chunk = sentence + " "
            else:
                current_chunk = para + "\n\n"
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def detect_gpu():
    """Detect if GPU is available and get basic information"""
    gpu_info = {"available": False, "name": "None", "memory": "0"}
    
    try:
        system = platform.system()
        if system == "Windows":
            # Use Windows Management Instrumentation
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        # Very simple extraction, would need more robust parsing for multiple GPUs
                        gpu_info["available"] = True
                        gpu_info["name"] = line.strip().split('  ')[0]
                        # Extract memory if available
                        if "AdapterRAM" in result.stdout:
                            try:
                                ram_bytes = int(re.search(r'\d+', line.split('  ')[-1]).group())
                                gpu_info["memory"] = f"{ram_bytes / (1024**3):.2f} GB"
                            except:
                                pass
        elif system == "Linux":
            # Try nvidia-smi for NVIDIA GPUs
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                gpu_info["available"] = True
                output = result.stdout.strip().split(',')
                if len(output) >= 2:
                    gpu_info["name"] = output[0].strip()
                    gpu_info["memory"] = output[1].strip()
        
        # Also try to detect via Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                response_json = response.json()
                # If Ollama is running, it means some level of hardware acceleration might be available
                gpu_info["ollama_running"] = True
        except:
            gpu_info["ollama_running"] = False
            
        return gpu_info
    except Exception as e:
        logger.error(f"Error detecting GPU: {str(e)}")
        return gpu_info

def process_text_with_llm(text, prompt, model="codellama:7b", gpu_options=None):
    """Process text using the local LLM with GPU acceleration"""
    # Create a system prompt for text processing
    system_prompt = """You are an expert document processor and text organizer.
    Your task is to take extracted text snippets from documents and convert them into:
    1. Well-structured paragraphs
    2. Coherent summaries
    3. Meaningful insights
    
    Maintain all the factual information from the original text.
    Format your response neatly with proper headings and sections where appropriate.
    """
    
    # Use the GPU options if provided
    if gpu_options is None:
        gpu_options = {
            "num_gpu": 1,
            "num_batch": 128
        }
    
    # Determine if text needs to be chunked
    text_length = len(text)
    logger.info(f"Processing text with LLM, text length: {text_length} characters")
    
    if text_length > 4000:
        logger.info("Text is long, using chunked processing")
        
        # Chunk the text
        text_chunks = chunk_text(text, max_chunk_size=3500)
        logger.info(f"Split text into {len(text_chunks)} chunks")
        
        # Process each chunk separately
        results = []
        
        for i, chunk in enumerate(text_chunks):
            chunk_prompt = f"{prompt} (Processing chunk {i+1} of {len(text_chunks)})"
            full_prompt = f"{chunk_prompt}\n\nHere is part of the extracted text to process:\n\n{chunk}"
            
            # Call the LLM through the API
            try:
                os.environ["OLLAMA_MODEL"] = model
                os.environ["OLLAMA_SYSTEM_PROMPT"] = system_prompt
                
                # Pass GPU options to generate_text
                chunk_result = generate_text(full_prompt, gpu_options=gpu_options)
                
                if isinstance(chunk_result, dict) and "error" in chunk_result:
                    logger.error(f"Error processing chunk {i+1}: {chunk_result['error']}")
                    # No retry with smaller context since we're not limiting context anymore
                
                if not isinstance(chunk_result, dict):
                    results.append(chunk_result)
                else:
                    results.append(f"Error processing chunk {i+1}: {chunk_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                results.append(f"Error processing chunk {i+1}: {str(e)}")
        
        # Combine the results
        if len(results) > 1:
            combined_result = "# Processed Document\n\n"
            for i, res in enumerate(results):
                combined_result += f"## Part {i+1}\n\n{res}\n\n"
                
            # Add a summary at the end
            combined_result += "\n\n## Overall Summary\n\nThis document was processed in multiple parts due to its length."
            return combined_result
        elif len(results) == 1:
            return results[0]
        else:
            return {"error": "Failed to process any chunks of the text"}
    else:
        # For shorter texts, process normally
        full_prompt = f"{prompt}\n\nHere is the extracted text to process:\n\n{text}"
        
        # Call the LLM through the API with GPU options
        try:
            os.environ["OLLAMA_MODEL"] = model
            os.environ["OLLAMA_SYSTEM_PROMPT"] = system_prompt
            
            result = generate_text(full_prompt, gpu_options=gpu_options)
            return result
        except Exception as e:
            logger.error(f"Error processing text with LLM: {str(e)}")
            return {"error": str(e)}

def save_extraction(document_name, original_text, processed_text):
    """Save extracted and processed text"""
    extraction_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    extraction_data = {
        "id": extraction_id,
        "document_name": document_name,
        "timestamp": timestamp,
        "original_text_length": len(original_text),
        "processed_text_length": len(processed_text)
    }
    
    # Save metadata
    with open(EXTRACTIONS_DIR / f"{extraction_id}_meta.json", 'w') as f:
        json.dump(extraction_data, f, indent=2)
    
    # Save original text
    with open(EXTRACTIONS_DIR / f"{extraction_id}_original.txt", 'w', encoding='utf-8') as f:
        f.write(original_text)
    
    # Save processed text
    with open(EXTRACTIONS_DIR / f"{extraction_id}_processed.txt", 'w', encoding='utf-8') as f:
        f.write(processed_text)
        
    return extraction_id

def get_saved_extractions():
    """Get list of all saved extractions"""
    extractions = []
    for file in EXTRACTIONS_DIR.glob("*_meta.json"):
        try:
            with open(file, 'r') as f:
                metadata = json.load(f)
                extractions.append(metadata)
        except Exception as e:
            logger.error(f"Error loading extraction metadata: {str(e)}")
    
    # Sort by timestamp, newest first
    extractions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return extractions

def run_forecasting(df, date_col, value_col, periods=30):
    """Run Prophet forecasting on selected columns."""
    if Prophet is None:
        return None, "Prophet library not installed. Install with: pip install prophet"
    try:
        df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
        df['ds'] = pd.to_datetime(df['ds'])
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast, None
    except Exception as e:
        return None, f"Forecasting error: {str(e)}"

def run_linear_regression(df, feature_col, target_col):
    """Run linear regression using scikit-learn."""
    if LinearRegression is None:
        return None, "scikit-learn not installed. Install with: pip install scikit-learn"
    try:
        X = np.array(df[feature_col]).reshape(-1, 1)
        y = df[target_col]
        model = LinearRegression().fit(X, y)
        return model, None
    except Exception as e:
        return None, f"Regression error: {str(e)}"

def show_document_processor():
    """Main document processor UI"""
    st.title("📄 Document Processing Assistant")
    
    # Add navigation back to main menu (if used in main app)
    if "current_page" in st.session_state:
        if st.button("← Back to Main Menu"):
            st.session_state.current_page = "main_menu"
            st.rerun()
    
    st.markdown("""
    This tool extracts text from your documents and helps you process it into 
    well-structured paragraphs and summaries, all using local AI.
    """)
    
    # Initialize session state for document management
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = {}  # {filename: text_content}
    
    # Initialize GPU settings in session state if not exists
    if "gpu_settings" not in st.session_state:
        gpu_info = detect_gpu()
        st.session_state.gpu_settings = {
            "use_gpu": gpu_info["available"],
            "num_gpu": 1,
            "gpu_layers": 43,  # Default for many models
            "use_f16": True    # Use half precision
        }
    
    # Add GPU settings in the sidebar
    with st.sidebar:
        st.header("GPU Settings")
        gpu_info = detect_gpu()
        
        if gpu_info["available"]:
            st.success(f"✅ GPU detected: {gpu_info['name']}")
            if "memory" in gpu_info and gpu_info["memory"] != "0":
                st.info(f"GPU Memory: {gpu_info['memory']}")
            
            # GPU options
            use_gpu = st.checkbox("Use GPU acceleration", value=st.session_state.gpu_settings["use_gpu"])
            
            if use_gpu:
                col1, col2 = st.columns(2)
                with col1:
                    num_gpu = st.number_input("Number of GPUs", 
                                             min_value=1, 
                                             max_value=8, 
                                             value=st.session_state.gpu_settings["num_gpu"])
                
                with col2:
                    gpu_layers = st.slider("GPU Layers", 
                                          min_value=1, 
                                          max_value=64, 
                                          value=st.session_state.gpu_settings["gpu_layers"],
                                          help="Number of layers to run on GPU. More layers use more GPU memory.")
                
                use_f16 = st.checkbox("Use half precision (FP16)", 
                                      value=st.session_state.gpu_settings["use_f16"],
                                      help="Reduces memory usage but may affect quality on some models")
                
                # Update session state
                st.session_state.gpu_settings = {
                    "use_gpu": use_gpu,
                    "num_gpu": num_gpu,
                    "gpu_layers": gpu_layers,
                    "use_f16": use_f16
                }
                
                # Show current settings
                st.info("Current GPU Settings:")
                st.code(f"""
                Number of GPUs: {num_gpu}
                GPU Layers: {gpu_layers}
                Half Precision: {"Enabled" if use_f16 else "Disabled"}
                """)
            else:
                st.session_state.gpu_settings["use_gpu"] = False
                st.info("GPU acceleration is disabled. Processing will use CPU only.")
        else:
            st.warning("⚠️ No GPU detected or required drivers not installed")
            st.session_state.gpu_settings["use_gpu"] = False
    
    # Set up tabs for upload/process, multi-document research, and view history
    tab1, tab2, tab3, tab4 = st.tabs([
        "Process Single Document", 
        "Multi-Document Research", 
        "Extraction History",
        "Economic Model"
    ])
    
    with tab1:
        st.subheader("Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document (PDF, DOCX, TXT, HTML, CSV, XLSX, XLS)", 
            type=["pdf", "docx", "txt", "html", "htm", "csv", "xlsx", "xls"],
            key="single_doc_uploader"
        )
        
        if uploaded_file:
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir) / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            file_ext = temp_path.suffix.lower()
            # Check if Excel/CSV
            if file_ext in [".csv", ".xlsx", ".xls"]:
                if file_ext == ".csv":
                    df = pd.read_csv(temp_path)
                    sheet_dfs = {"Sheet1": df}
                else:
                    # Read all sheets
                    excel_file = pd.ExcelFile(temp_path)
                    sheet_dfs = {sheet: excel_file.parse(sheet) for sheet in excel_file.sheet_names}
                st.success(f"Loaded spreadsheet: {uploaded_file.name}")
                # Show sheet selection if Excel
                if file_ext in [".xlsx", ".xls"]:
                    st.write("**Sheets found:**", list(sheet_dfs.keys()))
                    selected_sheet = st.selectbox("Select sheet to preview", list(sheet_dfs.keys()))
                    df = sheet_dfs[selected_sheet]
                else:
                    selected_sheet = "Sheet1"
                    df = sheet_dfs[selected_sheet]
                st.write(f"**Preview of '{selected_sheet}':**")
                st.dataframe(df.head())
                st.write("**Columns:**", list(df.columns))
                st.write("**Shape:**", df.shape)
                # Prompt for Power BI conversion
                st.subheader("Convert to Power BI Report")
                default_powerbi_prompt = (
                    "Analyze this spreadsheet and suggest how to convert it into a Power BI report. "
                    "Describe the main tables, relationships, and recommend visuals (charts, tables, KPIs). "
                    "If possible, suggest calculated columns or measures. "
                    "Also, consider the following user instructions:\n"
                )
                user_powerbi_prompt = st.text_area(
                    "Add instructions or describe what you want in the Power BI report",
                    placeholder="E.g. Focus on sales by region, add a time trend, highlight top 5 products, etc.",
                    height=80
                )
                if st.button("Analyze for Power BI", key="analyze_powerbi"):
                    with st.spinner("Analyzing spreadsheet for Power BI..."):
                        # Prepare a summary of all sheets for the LLM
                        if file_ext in [".xlsx", ".xls"]:
                            df_info = ""
                            for sheet_name, sdf in sheet_dfs.items():
                                df_info += f"\n\nSheet: {sheet_name}\nColumns: {list(sdf.columns)}\nShape: {sdf.shape}\nHead:\n{sdf.head().to_string()}\n"
                        else:
                            df_info = f"Columns: {list(df.columns)}\nShape: {df.shape}\nHead:\n{df.head().to_string()}"
                        llm_prompt = (
                            default_powerbi_prompt +
                            user_powerbi_prompt +
                            "\n\nSpreadsheet Info:\n" +
                            df_info
                        )
                        # Use the LLM to generate suggestions
                        model_options = ["llama4:latest", "codellama:7b", "qwen3:latest"]
                        selected_model = st.selectbox("Select LLM model", model_options, key="powerbi_model")
                        gpu_options = None
                        if st.session_state.gpu_settings["use_gpu"]:
                            gpu_options = {
                                "num_gpu": st.session_state.gpu_settings["num_gpu"],
                                "gpu_layers": st.session_state.gpu_settings["gpu_layers"],
                                "f16": st.session_state.gpu_settings["use_f16"]
                            }
                        result = process_text_with_llm(
                            llm_prompt,
                            "Suggest a Power BI report structure and visuals for this spreadsheet.",
                            selected_model,
                            gpu_options=gpu_options
                        )
                    if isinstance(result, dict) and "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("Power BI analysis complete!")
                        st.markdown(result)
                        st.download_button(
                            "Download Power BI Suggestions",
                            result,
                            f"{Path(uploaded_file.name).stem}_powerbi_suggestions.txt",
                            "text/plain"
                        )
            else:
                st.info(f"Uploaded: {uploaded_file.name}")
                # Extract text
                with st.spinner("Extracting text..."):
                    extracted_text = extract_text(temp_path)
                if extracted_text:
                    st.success("Text extracted successfully!")
                    with st.expander("View Extracted Text"):
                        st.text_area("Raw Text", extracted_text, height=200)
                    st.subheader("Process Text")
                    processing_prompt = st.text_area(
                        "Describe how you want the text processed",
                        value="Create well-structured paragraphs from these text snippets, organize related information, and provide a brief summary.",
                        height=100
                    )
                    model_options = ["llama4:latest", "codellama:7b", "qwen3:latest"]
                    selected_model = st.selectbox("Select LLM model", model_options)
                    if st.button("Process Text", type="primary"):
                        with st.spinner("Processing text with AI..."):
                            gpu_options = None
                            if st.session_state.gpu_settings["use_gpu"]:
                                gpu_options = {
                                    "num_gpu": st.session_state.gpu_settings["num_gpu"],
                                    "gpu_layers": st.session_state.gpu_settings["gpu_layers"],
                                    "f16": st.session_state.gpu_settings["use_f16"]
                                }
                            result = process_text_with_llm(
                                extracted_text, 
                                processing_prompt, 
                                selected_model,
                                gpu_options=gpu_options
                            )
                        if isinstance(result, dict) and "error" in result:
                            st.error(f"Error processing text: {result['error']}")
                        else:
                            st.success("Text processed successfully!")
                            extraction_id = save_extraction(uploaded_file.name, extracted_text, result)
                            st.subheader("Processed Text")
                            st.markdown(result)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "Download Processed Text",
                                    result,
                                    f"{Path(uploaded_file.name).stem}_processed.txt",
                                    "text/plain"
                                )
                            with col2:
                                st.download_button(
                                    "Download Original Text",
                                    extracted_text,
                                    f"{Path(uploaded_file.name).stem}_original.txt",
                                    "text/plain"
                                )
                else:
                    st.error("Failed to extract text from the document.")
    
    with tab2:
        st.subheader("Multi-Document Research")
        st.markdown("""
        Upload multiple documents and ask questions across all of them. 
        The AI will search through the documents and curate insights based on your query.
        """)
        
        # Multi-document uploader
        uploaded_files = st.file_uploader(
            "Upload multiple documents",
            type=["pdf", "docx", "txt", "html", "htm"],
            accept_multiple_files=True,
            key="multi_doc_uploader"
        )
        
        # Process newly uploaded files
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    if file.name not in st.session_state.uploaded_docs:
                        # Save the uploaded file temporarily
                        temp_dir = tempfile.mkdtemp()
                        temp_path = Path(temp_dir) / file.name
                        
                        with open(temp_path, "wb") as f:
                            f.write(file.getvalue())
                        
                        # Extract text from the file
                        extracted_text = extract_text(temp_path)
                        
                        # Store in session state
                        if extracted_text and not isinstance(extracted_text, str) or not extracted_text.startswith("Unsupported"):
                            st.session_state.uploaded_docs[file.name] = extracted_text
                            st.success(f"✅ Processed: {file.name}")
                        else:
                            st.error(f"❌ Failed to process: {file.name}")
        
        # Display current document set
        if st.session_state.uploaded_docs:
            st.write(f"📚 Current document set: {len(st.session_state.uploaded_docs)} documents")
            
            # Show document list
            with st.expander("View Document List"):
                for idx, (filename, text) in enumerate(st.session_state.uploaded_docs.items()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{idx+1}. **{filename}** ({len(text)} characters)")
                    with col2:
                        if st.button(f"Remove", key=f"remove_{idx}"):
                            del st.session_state.uploaded_docs[filename]
                            st.rerun()
            
            # Clear all documents button
            if st.button("Clear All Documents"):
                st.session_state.uploaded_docs = {}
                st.rerun()
            
            # Query interface
            st.subheader("Research Query")
            st.markdown("Ask questions about your documents or request specific analysis")
            
            research_prompt = st.text_area(
                "Enter your query or instructions",
                placeholder="Examples:\n- Find all mentions of budgets across these documents\n- Compare the key points made about climate change\n- Identify contradictions between these sources\n- Summarize the methodology sections from all papers",
                height=100
            )
            
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                model_options = ["llama4:latest", "codellama:7b", "qwen3:latest"]
                selected_model = st.selectbox("Select LLM model", model_options, key="multi_model")
            
            with col2:
                max_tokens = st.slider("Response Length", min_value=500, max_value=8000, value=2000, step=500)
            
            # Process query button
            if st.button("Process Query", type="primary") and research_prompt and st.session_state.uploaded_docs:
                # Create a batch ID for this research
                batch_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Get total text size
                total_text_size = sum(len(text) for text in st.session_state.uploaded_docs.values())
                st.info(f"Processing {len(st.session_state.uploaded_docs)} documents with total size of {total_text_size} characters")
                
                # Determine memory-efficient processing strategy
                if total_text_size > 15000:
                    st.warning("Large document set detected. Using memory-efficient processing mode.")
                    
                    # Process each document separately with the same query
                    all_results = []
                    progress_bar = st.progress(0)
                    
                    for i, (filename, doc_text) in enumerate(st.session_state.uploaded_docs.items()):
                        st.write(f"Processing document {i+1}/{len(st.session_state.uploaded_docs)}: {filename}")
                        
                        # Create a focused prompt for this document
                        doc_query = f"""
                        RESEARCH QUERY: {research_prompt}
                        
                        DOCUMENT: {filename}
                        
                        {doc_text[:10000]}  # Limit individual document size
                        """
                        
                        # Set system prompt
                        research_system_prompt = """You are analyzing a single document as part of a multi-document research task.
                        Focus on extracting key information relevant to the research query.
                        Format your response with clear headings.
                        Include document name when citing information.
                        """
                        
                        os.environ["OLLAMA_MODEL"] = selected_model
                        os.environ["OLLAMA_SYSTEM_PROMPT"] = research_system_prompt
                        
                        # Process this document
                        doc_result = process_text_with_llm(
                            doc_query,
                            f"Analyze this document ({filename}) for information relevant to: {research_prompt}",
                            selected_model
                        )
                        
                        if not isinstance(doc_result, dict):
                            all_results.append(f"## Analysis of {filename}\n\n{doc_result}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(st.session_state.uploaded_docs))
                    
                    # Now create a synthesis prompt with the individual analyses
                    synthesis_text = "\n\n".join(all_results)
                    
                    synthesis_prompt = f"""
                    RESEARCH QUERY: {research_prompt}
                    
                    Below are analyses of individual documents. Create a coherent synthesis that answers the research query.
                    
                    {synthesis_text[:20000]}  # Limit size for final synthesis
                    """
                    
                    # Set synthesis system prompt
                    synthesis_system_prompt = """You are creating a final synthesis of multiple document analyses.
                    Your task is to integrate the findings from all documents into a coherent response to the research query.
                    Organize your response with clear headings and structure.
                    Highlight agreements and contradictions between sources.
                    Include a summary section at the end.
                    """
                    
                    os.environ["OLLAMA_MODEL"] = selected_model
                    os.environ["OLLAMA_SYSTEM_PROMPT"] = synthesis_system_prompt
                    
                    with st.spinner("Creating final synthesis..."):
                        # Remove context_length parameter
                        result = generate_text(synthesis_prompt)
                else:
                    # Original processing for smaller document sets
                    # Combine all document texts with headers
                    combined_text = ""
                    for filename, text in st.session_state.uploaded_docs.items():
                        combined_text += f"\n\n=== DOCUMENT: {filename} ===\n\n{text}"
                    
                    # Create a system prompt for research
                    research_system_prompt = """You are an expert research assistant.
                    You have been provided with multiple documents to analyze.
                    Your task is to search through these documents to find relevant information and provide a coherent response.
                    Always cite the document name when referencing specific information.
                    Organize your response with clear headings and structure.
                    Include direct quotes when appropriate, properly attributed to their source document.
                    """
                    
                    # Set the environment variables
                    os.environ["OLLAMA_MODEL"] = selected_model
                    os.environ["OLLAMA_SYSTEM_PROMPT"] = research_system_prompt
                    
                    # Create the full query
                    full_query = f"""
                    RESEARCH QUERY: {research_prompt}
                    
                    DOCUMENTS PROVIDED:
                    {combined_text[:20000]}  # Reduced from 50000 to save memory
                    """
                    
                    with st.spinner("Analyzing documents and generating response..."):
                        # Call the LLM with reduced context size
                        result = process_text_with_llm(
                            full_query, 
                            "Please analyze the provided documents and answer the query.", 
                            selected_model
                        )
                
                # Display the result
                if isinstance(result, dict) and "error" in result:
                    st.error(f"Error processing query: {result['error']}")
                else:
                    st.success("Analysis complete!")
                    
                    # Save the batch
                    batch_data = {
                        "id": batch_id,
                        "timestamp": timestamp,
                        "query": research_prompt,
                        "documents": list(st.session_state.uploaded_docs.keys()),
                        "model": selected_model
                    }
                    
                    with open(BATCH_DIR / f"{batch_id}_meta.json", 'w') as f:
                        json.dump(batch_data, f, indent=2)
                    
                    with open(BATCH_DIR / f"{batch_id}_result.txt", 'w', encoding='utf-8') as f:
                        f.write(result)
                    
                    # Display the response
                    st.subheader("Research Results")
                    st.markdown(result)
                    
                    # Download option
                    st.download_button(
                        "Download Results",
                        result,
                        f"research_results_{timestamp}.txt",
                        "text/plain"
                    )
            
            # Show previous research results
            st.subheader("Previous Research")
            
            # Get batch files
            batch_files = list(BATCH_DIR.glob("*_meta.json"))
            if batch_files:
                batches = []
                for file in batch_files:
                    try:
                        with open(file, 'r') as f:
                            batch_data = json.load(f)
                            batches.append(batch_data)
                    except Exception as e:
                        logger.error(f"Error loading batch metadata: {str(e)}")
                
                # Sort by timestamp, newest first
                batches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                # Display as selectable list
                selected_batch = st.selectbox(
                    "Select previous research",
                    options=[f"{b['timestamp']} - {b['query'][:50]}..." for b in batches],
                    index=0 if batches else None
                )
                
                if selected_batch and batches:
                    # Extract batch ID from selection
                    selected_timestamp = selected_batch.split(" - ")[0]
                    selected_batch_data = next((b for b in batches if b["timestamp"] == selected_timestamp), None)
                    
                    if selected_batch_data:
                        batch_id = selected_batch_data["id"]
                        result_path = BATCH_DIR / f"{batch_id}_result.txt"
                        
                        if result_path.exists():
                            with open(result_path, 'r', encoding='utf-8') as f:
                                result_text = f.read()
                            
                            # Display the result
                            st.markdown(result_text)
                            
                            # Show document list
                            st.write("Documents used:")
                            for doc in selected_batch_data.get("documents", []):
                                st.write(f"- {doc}")
            else:
                st.info("No previous research found.")
    
    with tab3:
        st.subheader("Extraction History")
        
        extractions = get_saved_extractions()
        
        if not extractions:
            st.info("No extraction history found.")
        else:
            st.write(f"Found {len(extractions)} previous extractions")
            
            # Convert to DataFrame for display
            df = pd.DataFrame([{
                "Date": x.get("timestamp", "").replace("_", " "),
                "Document": x.get("document_name", "Unknown"),
                "ID": x.get("id", "")
            } for x in extractions])
            
            st.dataframe(df)
            
            # Select extraction to view
            selected_id = st.selectbox("Select extraction to view", 
                                      [x.get("id") for x in extractions],
                                      format_func=lambda x: next((e.get("document_name") for e in extractions if e.get("id") == x), x))
            
            if selected_id:
                # Load selected extraction
                try:
                    # Load original text
                    original_path = EXTRACTIONS_DIR / f"{selected_id}_original.txt"
                    with open(original_path, 'r', encoding='utf-8') as f:
                        original_text = f.read()
                        
                    # Load processed text
                    processed_path = EXTRACTIONS_DIR / f"{selected_id}_processed.txt"
                    with open(processed_path, 'r', encoding='utf-8') as f:
                        processed_text = f.read()
                    
                    # Display the texts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("Original Text", expanded=False):
                            st.text_area("", original_text, height=400)
                            
                    with col2:
                        st.subheader("Processed Text")
                        st.markdown(processed_text)
                        
                    # Download buttons
                    st.download_button(
                        "Download Processed Text",
                        processed_text,
                        f"extraction_{selected_id}_processed.txt",
                        "text/plain"
                    )
                except Exception as e:
                    st.error(f"Error loading extraction: {str(e)}")
    
    with tab4:
        st.subheader("Economic Model")
        st.markdown("""
        Upload a CSV or Excel file to develop economic models for energy projects.
        You can also enter key parameters for economic evaluation of energy investments.
        """)
        uploaded_data_file = st.file_uploader(
            "Upload CSV/Excel for Economic Modeling", 
            type=["csv", "xls", "xlsx"],
            key="economic_data_uploader"
        )
        df = None
        data_preview = ""
        if uploaded_data_file:
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir) / uploaded_data_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_data_file.getvalue())
            file_ext = temp_path.suffix.lower()
            if file_ext == ".csv":
                df = pd.read_csv(temp_path)
            elif file_ext in [".xls", ".xlsx"]:
                df = pd.read_excel(temp_path)
            if df is not None:
                st.write("Data Preview:")
                st.dataframe(df.head())
                data_preview = df.head().to_string()
            else:
                st.error("Failed to load data file.")

        # Create a 2-column layout for Economic Model and Chat
        econ_col, chat_col = st.columns([3, 2])
        
        # Initialize chat history in session state if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hello! I'm the Baby Todd Energy Assistant. I can help with your economic model questions. What would you like to know about energy economics?"}
            ]

        with econ_col:
            # Economic model parameters
            st.markdown("#### Economic Parameters")
            col1, col2 = st.columns(2)
            with col1:
                capital_expenditure = st.number_input("Capital Expenditure ($)", min_value=0, value=1000000)
                annual_revenue = st.number_input("Annual Revenue ($)", min_value=0, value=500000)
                operating_costs = st.number_input("Annual Operating Costs ($)", min_value=0, value=200000)
            with col2:
                discount_rate = st.slider("Discount Rate (%)", min_value=0, max_value=25, value=10)
                project_life = st.slider("Project Life, End of Field life (years) ", min_value=1, max_value=30, value=10)
                tax_rate = st.slider("Tax Rate (%)", min_value=0, max_value=50, value=21)

            if st.button("Calculate Economic Metrics"):
                st.markdown("#### Economic Analysis Results")
                
                # Simple NPV calculation
                cash_flows = [-capital_expenditure]
                annual_profit = annual_revenue - operating_costs
                annual_after_tax = annual_profit * (1 - tax_rate/100)
                
                for year in range(1, project_life + 1):
                    cash_flows.append(annual_after_tax)
                
                # Calculate NPV
                npv = 0
                for t, cf in enumerate(cash_flows):
                    npv += cf / ((1 + discount_rate/100) ** t)
                
                # Calculate simple payback
                if annual_after_tax > 0:
                    simple_payback = capital_expenditure / annual_after_tax
                else:
                    simple_payback = float('inf')
                
                # Calculate IRR (simple approach)
                try:
                    irr = np.irr(cash_flows) * 100
                except:
                    irr = "Could not calculate"
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Net Present Value (NPV)", f"${npv:,.2f}")
                    st.metric("Internal Rate of Return (IRR)", f"{irr}%" if isinstance(irr, (int, float)) else irr)
                with col2:
                    st.metric("Payback Period (years)", f"{simple_payback:.2f}")
                    st.metric("Profit-Investment Ratio", f"{npv/capital_expenditure:.2f}" if capital_expenditure > 0 else "N/A")
                
                # Create a simple cash flow chart
                years = list(range(project_life + 1))
                cumulative_cf = [sum(cash_flows[:i+1]) for i in range(len(cash_flows))]
                
                cf_df = pd.DataFrame({
                    'Year': years,
                    'Cash Flow': cash_flows,
                    'Cumulative Cash Flow': cumulative_cf
                })
                
                st.write("### Cash Flow Analysis")
                st.line_chart(cf_df.set_index('Year'))
                st.write(cf_df)

                # Store calculated metrics in session state for the chat to access
                st.session_state.economic_metrics = {
                    "capex": capital_expenditure,
                    "annual_revenue": annual_revenue,
                    "operating_costs": operating_costs,
                    "discount_rate": discount_rate,
                    "project_life": project_life,
                    "tax_rate": tax_rate,
                    "npv": npv,
                    "irr": irr,
                    "simple_payback": simple_payback,
                    "profit_investment_ratio": npv/capital_expenditure if capital_expenditure > 0 else "N/A",
                    "cash_flows": cash_flows,
                    "cumulative_cf": cumulative_cf
                }

                # Energy-specific metrics
                if df is not None:
                    st.markdown("#### Project-Specific Analysis")
                    st.info("Upload project-specific data to enable detailed economic analysis")
                    
                    # Placeholder for more advanced analysis based on uploaded data
                    model_options = ["Energy Asset Valuation", "Carbon Credit Analysis", "Renewable ROI Comparison"]
                    selected_model = st.selectbox("Select economic model type", model_options)
                    
                    if st.button("Run Advanced Analysis"):
                        with st.spinner("Running economic analysis..."):
                            st.success("Analysis complete!")
                            st.write("This would show detailed economic analysis based on the selected model type.")

        # Chat interface in the right column
        with chat_col:
            st.markdown("#### Baby Todd Energy Assistant Chat")
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.markdown(f"**You**: {message['content']}")
                    else:
                        st.markdown(f"**Baby Todd Energy Assistant**: {message['content']}")
                st.divider()
            
            # Handle suggested questions first
            if "suggested_question" in st.session_state:
                # Pre-fill the input with the suggested question
                user_input = st.text_input(
                    "Ask about the economic model:", 
                    value=st.session_state.suggested_question,
                    key="user_message"
                )
                # Remove the suggestion after it's been used
                del st.session_state.suggested_question
            else:
                # Regular empty input if no suggestion
                user_input = st.text_input(
                    "Ask about the economic model:",
                    key="user_message"
                )
                
            if st.button("Send", key="send_message") or user_input:
                if user_input:
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Generate response based on economic metrics if available
                    if "economic_metrics" in st.session_state:
                        metrics = st.session_state.economic_metrics
                        
                        # Prepare system prompt with current economic data
                        system_prompt = f"""You are Baby Todd Energy Assistant, an expert in energy economics.
                        The user has built an economic model with the following parameters:
                        - Capital Expenditure: ${metrics['capex']:,}
                        - Annual Revenue: ${metrics['annual_revenue']:,}
                        - Annual Operating Costs: ${metrics['operating_costs']:,}
                        - Discount Rate: {metrics['discount_rate']}%
                        - Project Life: {metrics['project_life']} years
                        - Tax Rate: {metrics['tax_rate']}%
                        
                        The calculated metrics are:
                        - Net Present Value (NPV): ${metrics['npv']:,.2f}
                        - IRR: {metrics['irr']}%
                        - Payback Period: {metrics['simple_payback']:.2f} years
                        
                        Answer the user's question about this economic model. Provide insights about the economic 
                        viability of this energy project based on these numbers. Be concise but helpful.
                        """
                        
                        os.environ["OLLAMA_SYSTEM_PROMPT"] = system_prompt
                        try:
                            response = generate_text(user_input, model="codellama:7b")
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.session_state.chat_history.append({"role": "assistant", "content": f"I'm sorry, I encountered an error: {str(e)}"})
                    else:
                        # If no economic metrics have been calculated yet
                        response = "I don't see any economic metrics calculated yet. Please fill in the economic parameters on the left and click 'Calculate Economic Metrics' first. Then I can help analyze the results."
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Rerun to update the chat display
                    st.rerun()

            # Add buttons for suggested questions
            st.write("Suggested questions:")
            suggest_col1, suggest_col2 = st.columns(2)
            with suggest_col1:
                if st.button("Is this project profitable?"):
                    # Store the suggested question in session state
                    st.session_state.suggested_question = "Is this project profitable?"
                    st.rerun()
            with suggest_col2:
                if st.button("How can I improve the IRR?"):
                    # Store the suggested question in session state
                    st.session_state.suggested_question = "How can I improve the IRR?"
                    st.rerun()

# Run standalone if executed directly
if __name__ == "__main__":
    show_document_processor()