import streamlit as st
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
import tempfile
import json
import matplotlib.pyplot as plt
from datetime import datetime
import uuid

# Add parent directory to path for importing ollama utilities
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
# Fix the import - import only what's available
from api import generate_text  # Import only the available function

# Create a local Config class to handle configuration
class Config:
    """Central configuration for API settings and defaults"""
    # Base URLs
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Model defaults
    DEFAULT_TEXT_MODEL = os.environ.get("DEFAULT_TEXT_MODEL", "qwen3:latest")
    DEFAULT_CODE_MODEL = os.environ.get("DEFAULT_CODE_MODEL", "qwen3:latest")
    
    # API endpoints
    @classmethod
    def tags_url(cls):
        return f"{cls.OLLAMA_BASE_URL}/api/tags"
    
    @classmethod
    def generate_url(cls):
        return f"{cls.OLLAMA_BASE_URL}/api/generate"

# Define function to directly call Ollama API with model parameter
def generate_text_with_model(prompt, model, system_prompt=None):
    """Generate text using Ollama API with specified model"""
    try:
        import requests
        
        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        # Make the API request
        response = requests.post(Config.generate_url(), json=payload)
        
        if response.status_code != 200:
            return f"Error: API request failed with status code {response.status_code}"
        
        # Extract the response
        result = response.json()
        response_text = result.get("response", "")
        
        # Remove thinking tags from the response
        import re
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        return response_text
    
    except Exception as e:
        return f"Error: {str(e)}"

# Define the get_available_ollama_models function locally
def get_available_ollama_models():
    """Fetch available Ollama models using the API"""
    try:
        import requests
        response = requests.get(Config.tags_url())
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        else:
            return []
    except Exception as e:
        print(f"Error fetching Ollama models: {str(e)}")
        return []

# Data storage paths
DATA_DIR = parent_dir / "data"
MODELS_DIR = DATA_DIR / "economic_models"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# Only set page config when running this script directly (not when imported)
if __name__ == "__main__":
    st.set_page_config(page_title="Energy Economic Model", layout="wide")

def calculate_npv(cash_flows, discount_rate):
    """Calculate Net Present Value of a series of cash flows"""
    npv = 0
    for t, cf in enumerate(cash_flows):
        npv += cf / ((1 + discount_rate/100) ** t)
    return npv

def calculate_irr(cash_flows):
    """Calculate Internal Rate of Return"""
    try:
        irr = np.irr(cash_flows) * 100
        return irr
    except:
        return None

def calculate_payback(initial_investment, annual_cash_flow):
    """Calculate simple payback period"""
    if annual_cash_flow <= 0:
        return float('inf')
    return initial_investment / annual_cash_flow

def generate_cash_flows(capex, annual_revenue, operating_costs, tax_rate, project_life):
    """Generate cash flow projections for the project life"""
    cash_flows = [-capex]
    annual_profit = annual_revenue - operating_costs
    annual_after_tax = annual_profit * (1 - tax_rate/100)
    
    for year in range(1, project_life + 1):
        cash_flows.append(annual_after_tax)
    
    return cash_flows

def save_economic_model(model_data):
    """Save economic model data to file"""
    model_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_data["id"] = model_id
    model_data["timestamp"] = timestamp
    
    with open(MODELS_DIR / f"{model_id}.json", 'w') as f:
        json.dump(model_data, f, indent=2)
    
    return model_id

def load_economic_models():
    """Load all saved economic models"""
    models = []
    for file in MODELS_DIR.glob("*.json"):
        try:
            with open(file, 'r') as f:
                model_data = json.load(f)
                models.append(model_data)
        except Exception as e:
            print(f"Error loading model {file}: {str(e)}")
    
    # Sort by timestamp, newest first
    models.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return models

def show_economic_model():
    """Main function that displays the economic model interface"""
    st.title("🔬 Energy Economic Model & ToddLer Energy Assistant")
    
    # Add back to main menu button - only works when running in the main app
    if 'current_page' in st.session_state:
        if st.button("← Back to Main Menu"):
            st.session_state.current_page = "main_menu"
            st.rerun()
    
    # Add sidebar for Ollama configuration
    with st.sidebar:
        st.subheader("Ollama")
        st.markdown("""
        This app requires [Ollama](https://ollama.com/download).
        
        If you're seeing connection errors:
        1. Make sure Ollama is running
        2. Make sure you have the selected model installed
        """)
        
        # Add Connection Check Button
        if st.button("Check Ollama Connection"):
            try:
                import requests
                response = requests.get(Config.tags_url())
                if response.status_code == 200:
                    available_models = [model["name"] for model in response.json()["models"]]
                    current_model = st.session_state.get("selected_model", Config.DEFAULT_TEXT_MODEL)
                    if current_model in available_models:
                        st.success(f"✅ Connected to Ollama and {current_model} model is available")
                    else:
                        st.warning(f"⚠️ Connected to Ollama but {current_model} model is not available. Available models: {', '.join(available_models)}")
                else:
                    st.error(f"❌ Failed to fetch models from Ollama: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Failed to connect to Ollama: {str(e)}")
                st.info("Make sure Ollama is running. Download from https://ollama.com/download")
        
        # Add Advanced Model Settings expander
        with st.expander("Advanced Model Settings"):
            # Initialize selected model in session state if not exists
            if "selected_model" not in st.session_state:
                st.session_state.selected_model = Config.DEFAULT_TEXT_MODEL
                
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
                
                # Add "Test Selected Model" button
                if st.button("🧪 Test Selected Model"):
                    with st.spinner(f"Testing model: {selected_model}"):
                        test_prompt = "Say hello and introduce yourself in one sentence."
                        # Add instructions to not use thinking tags
                        test_system = """You are a helpful AI assistant that specializes in energy economics. 
                        Do not include your thinking process. Do not use <think> or </think> tags in your response."""
                        
                        try:
                            # Test the model with our direct API call
                            response = generate_text_with_model(test_prompt, selected_model, test_system)
                            
                            if response and not response.startswith("Error:"):
                                st.success(f"✅ Model test successful! Response: {response}")
                            else:
                                st.error(f"❌ Model test failed: {response}")
                        except Exception as e:
                            st.error(f"❌ Error testing model: {str(e)}")
            else:
                # Fallback to text input if no models are found
                st.warning("⚠️ No Ollama models found. Please check your Ollama installation.")
                selected_model = st.text_input(
                    "Ollama Model",
                    value=st.session_state.selected_model,
                    help="Name of the Ollama model to use (e.g., qwen3, llama3, mistral)"
                )
                
                if selected_model != st.session_state.selected_model:
                    st.session_state.selected_model = selected_model
                
                st.info("After installing new models, click 'Refresh' to update the list.")
            
            # System prompt configuration
            st.subheader("System Prompt")
            
            default_system_prompt = """You are ToddLer, an energy economics assistant. You help analyze economic models for energy projects.
            Be concise, helpful, and accurate in your responses. Use the economic context provided, but don't just repeat numbers.
            Provide insights, explanations, and recommendations when appropriate.
            """
            
            if "system_prompt" not in st.session_state:
                st.session_state.system_prompt = default_system_prompt
                
            custom_system_prompt = st.text_area(
                "Custom System Prompt",
                value=st.session_state.system_prompt,
                height=200,
                help="Customize the system prompt sent to the model to guide the responses"
            )
            
            if custom_system_prompt != st.session_state.system_prompt:
                st.session_state.system_prompt = custom_system_prompt
    
    st.markdown(
        """
        Upload your energy project spreadsheet, enter your project's carbon and gas numbers, set project life, 
        and let ToddLer help you analyze!
        """
    )

    tab1, tab2 = st.columns([3, 2])

    # === LEFT: Economic Model ===
    with tab1:
        st.header("Project Data & Economic Model")

        data_file = st.file_uploader(
            "Upload Project Data (Excel or CSV - optional):",
            type=["xlsx", "xls", "csv"]
        )

        df = None
        if data_file:
            file_ext = Path(data_file.name).suffix.lower()
            if file_ext == ".csv":
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
        else:
            st.info("You can upload a spreadsheet, but it's optional.")

        st.markdown("### Economic Parameters")
        col1, col2 = st.columns(2)
        with col1:
            capital_expenditure = st.number_input("Capital Expenditure ($)", min_value=0, value=1000000)
            annual_revenue = st.number_input("Annual Revenue ($)", min_value=0, value=500000)
            operating_costs = st.number_input("Annual Operating Costs ($)", min_value=0, value=200000)
            # Add new input fields for carbon and gas
            carbon = st.number_input("Total Carbon (tCO₂)", min_value=0.0, value=1000.0, step=100.0, 
                                    help="Total carbon emissions in tonnes of CO₂ over the project lifetime")
            gas = st.number_input("Total Gas (TJ)", min_value=0.0, value=500.0, step=50.0,
                                 help="Total gas production/consumption in Terajoules over the project lifetime")
        with col2:
            discount_rate = st.slider("Discount Rate (%)", min_value=0, max_value=25, value=10)
            project_life = st.slider("Project Life (years)", min_value=1, max_value=30, value=10)
            tax_rate = st.slider("Tax Rate (%)", min_value=0, max_value=50, value=21)
            # Add field for end of field life
            end_of_field_life = st.number_input("End of Field Life (years)", min_value=1, max_value=50, 
                                               value=project_life, help="Expected end of field life, typically equal to or greater than project life")

        # Add project details
        with st.expander("Project Details", expanded=False):
            project_name = st.text_input("Project Name", value="Energy Project")
            project_description = st.text_area("Project Description", value="", placeholder="Enter project description...")
            project_location = st.text_input("Project Location", value="")
            project_type = st.selectbox("Project Type", 
                                       ["Solar", "Wind", "Oil & Gas", "Hydro", "Geothermal", "Battery Storage", "Other"])
            if project_type == "Other":
                project_type_other = st.text_input("Specify Project Type")
                if project_type_other:
                    project_type = project_type_other

        st.markdown("----")

        # Use the input values for carbon and gas instead of default values
        if st.button("Calculate Metrics"):
            st.success("Calculation complete!")

            # Generate cash flows
            cash_flows = generate_cash_flows(capital_expenditure, annual_revenue, operating_costs, tax_rate, project_life)
            
            # Calculate economic metrics
            npv = calculate_npv(cash_flows, discount_rate)
            irr = calculate_irr(cash_flows)
            if annual_revenue - operating_costs > 0:
                simple_payback = calculate_payback(capital_expenditure, annual_revenue - operating_costs)
            else:
                simple_payback = float('inf')
            
            # Results DataFrame: use uploaded data or just make a summary table
            summary_data = {
                "Metric": [
                    "Total Carbon (tCO₂)",
                    "Total Gas (TJ)",
                    "Project Life (years)",
                    "End of Field Life (years)",
                    "Carbon per Year (tCO₂/yr)",
                    "Gas per Year (TJ/yr)",
                    "Net Present Value (NPV)",
                    "Internal Rate of Return (IRR)",
                    "Payback Period (years)",
                    "Profit-Investment Ratio",
                ],
                "Value": [
                    carbon,
                    gas,
                    project_life,
                    end_of_field_life,
                    carbon / project_life if project_life else None,
                    gas / project_life if project_life else None,
                    npv,
                    irr,
                    simple_payback,
                    npv/capital_expenditure if capital_expenditure > 0 else None,
                ]
            }
            summary_df = pd.DataFrame(summary_data)

            st.subheader("📊 Project Metrics")
            st.dataframe(summary_df)

            # If the user uploaded a spreadsheet, we can also append calculated columns
            if df is not None:
                # Example: add per-year columns
                df["Carbon per Year (tCO₂/yr)"] = carbon / project_life if project_life else None
                df["Gas per Year (TJ/yr)"] = gas / project_life if project_life else None
                st.subheader("Your Data with Calculations:")
                st.dataframe(df)

                # Download processed file
                csv = df.to_csv(index=False).encode()
                st.download_button("Download Results as CSV", data=csv, file_name="economic_results.csv", mime="text/csv")
            else:
                # Download just the summary table
                csv = summary_df.to_csv(index=False).encode()
                st.download_button("Download Metrics as CSV", data=csv, file_name="project_metrics.csv", mime="text/csv")
        st.markdown("----")

    # === RIGHT: Baby Todd Energy Assistant ===
    with tab2:
        st.header("💬 ToddLer Energy Assistant")

        # Create a chat-like interface with custom CSS
        st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .user-message {
            align-self: flex-end;
            background-color: #0084ff;
            color: white;
            border-radius: 18px;
            padding: 8px 12px;
            max-width: 80%;
            margin-left: auto;
        }
        .assistant-message {
            align-self: flex-start;
            background-color: #f0f0f0;
            color: black;
            border-radius: 18px;
            padding: 8px 12px;
            max-width: 80%;
            margin-right: auto;
        }
        </style>
        """, unsafe_allow_html=True)

        # Start chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hello! I'm ToddLer, your energy economics buddy. Ask me anything about your project numbers or economic analysis!"}
            ]

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # End chat container
        st.markdown('</div>', unsafe_allow_html=True)

        # Input area with send button side by side
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("", placeholder="Ask about the economic model...")
        with col2:
            send_button = st.button("Send")

        # Fix the error in the check for Enter key in session_state
        if send_button or (user_input and user_input.strip() and 
                          "Enter" in st.session_state and st.session_state.Enter):
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Create context for the AI with current project parameters
            context = f"""
            Economic Model Parameters:
            - Capital Expenditure: ${capital_expenditure:,.2f}
            - Annual Revenue: ${annual_revenue:,.2f}
            - Annual Operating Costs: ${operating_costs:,.2f}
            - Discount Rate: {discount_rate}%
            - Project Life: {project_life} years
            - Tax Rate: {tax_rate}%
            - Total Carbon: {carbon} tCO₂
            - Total Gas: {gas} TJ
            - End of Field Life: {end_of_field_life} years
            """

            # Use the custom system prompt from the sidebar
            system_prompt = st.session_state.system_prompt
            
            # Add to system prompt to remove thinking tags
            system_prompt += "\nDo not include your thinking process. Do not use <think> or </think> tags in your response."
            
            # Generate response using Ollama with the selected model
            try:
                with st.spinner("ToddLer is thinking..."):
                    # Use the selected model from session state
                    model = st.session_state.selected_model
                    
                    # Format prompt with the user's question
                    prompt = f"{context}\n\nUser question: {user_input}\n\nToddLer's response:"
                    
                    # Use our custom function that accepts the model parameter directly
                    answer = generate_text_with_model(prompt, model, system_prompt)
                    
                    # Debug the response
                    if not answer:
                        st.warning("Received empty response from Ollama")
                        raise Exception("Empty response from Ollama API")
                    elif len(answer.strip()) < 5:
                        st.warning(f"Received very short response: '{answer}'")
                        raise Exception("Response too short")
                    elif answer.startswith("Error:"):
                        raise Exception(answer)
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                
                # Show more detailed error information
                with st.expander("Error Details"):
                    st.error(f"Error connecting to Ollama: {str(e)}")
                    st.code(error_details)
                    
                    # Check if Ollama is running
                    try:
                        import requests
                        st.write("Checking Ollama connection...")
                        response = requests.get(Config.tags_url(), timeout=2)
                        if response.status_code == 200:
                            models = [m["name"] for m in response.json().get("models", [])]
                            st.write(f"Ollama is running. Available models: {', '.join(models)}")
                            if model not in models:
                                st.warning(f"The selected model '{model}' is not available. Try one of the available models.")
                        else:
                            st.error(f"Ollama returned status code {response.status_code}")
                    except Exception as conn_err:
                        st.error(f"Failed to connect to Ollama: {str(conn_err)}")
                        st.info("Make sure Ollama is running on http://localhost:11434")
                
                # Fallback to canned responses with more informative error message
                answer = f"I'm having trouble connecting to my brain right now. Let me use my backup circuits! (Error: {str(e)})"
                
                # Basic canned responses as fallback
                if "carbon" in user_input.lower():
                    answer = f"Your project's total carbon is {carbon:.2f} tCO₂, which averages {carbon / project_life if project_life else 0:.2f} tCO₂ per year over the project life."
                elif "gas" in user_input.lower():
                    answer = f"Your project's total gas is {gas:.2f} TJ, or {gas / project_life if project_life else 0:.2f} TJ per year."
                elif "profitable" in user_input.lower():
                    answer = "Profitability depends on revenue and cost. Based on your numbers, you should evaluate the NPV and IRR against your company's minimum thresholds."
                elif "summary" in user_input.lower():
                    answer = f"Project summary:\n- Capital Expenditure: ${capital_expenditure:,.2f}\n- Annual Revenue: ${annual_revenue:,.2f}\n- Carbon: {carbon:.2f} tCO₂\n- Gas: {gas:.2f} TJ\n- Project Life: {project_life} years"

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        # Add keyboard shortcut for Enter key
        st.markdown("""
        <script>
        const doc = window.parent.document;
        const inputs = doc.querySelectorAll('input[type="text"]');
        
        // Find the correct input (chat input)
        const chatInput = Array.from(inputs).find(input => 
            input.placeholder && input.placeholder.includes('Ask about the economic model'));
            
        if (chatInput) {
            chatInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    
                    // Find and click the send button
                    const buttons = doc.querySelectorAll('button');
                    const sendButton = Array.from(buttons).find(btn => 
                        btn.innerText === 'Send');
                        
                    if (sendButton) {
                        sendButton.click();
                    }
                }
            });
        }
        </script>
        """, unsafe_allow_html=True)

    # Saved models section
    st.header("Saved Economic Models")
    models = load_economic_models()
    
    if not models:
        st.info("No saved economic models found.")
    else:
        selected_model = st.selectbox(
            "Select a saved model",
            options=[f"{m.get('timestamp', 'Unknown')} - {m.get('project_name', 'Unnamed Project')}" for m in models],
            index=0
        )
        
        selected_idx = [f"{m.get('timestamp', 'Unknown')} - {m.get('project_name', 'Unnamed Project')}" for m in models].index(selected_model)
        model_data = models[selected_idx]
        
        st.subheader(f"Model: {model_data.get('project_name', 'Unnamed Project')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Project Type:** {model_data.get('project_type', 'N/A')}")
            st.write(f"**Location:** {model_data.get('project_location', 'N/A')}")
            st.write(f"**Date Created:** {model_data.get('timestamp', 'Unknown')}")
        with col2:
            st.metric("NPV", f"${model_data.get('npv', 0):,.2f}")
            st.metric("IRR", f"{model_data.get('irr', 0):.2f}%")
        
        st.write("**Description:**")
        st.write(model_data.get('project_description', 'No description available'))

        # Option to load this model's parameters
        if st.button("Load Parameters from this Model"):
            # Set session state values to match the selected model
            st.session_state.economic_metrics = model_data
            # Rerun to update the UI with these values
            st.rerun()

def main():
    show_economic_model()

# Fix the duplicate main() function calls
if __name__ == "__main__":
    main()
