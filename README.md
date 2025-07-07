# Local AI Assistant & Economic Modeling Tools

A comprehensive local AI-powered application suite featuring economic modeling, text processing, and multi-tool AI capabilities - all running locally without cloud dependencies.

![Local AI Assistant](https://patrickcastr/toddler/main/docs/images/preview.png)

## 🚀 Features

### 🧠 Local AI Processing
- **Multiple Models**: Support for various Ollama models (llama3.2, qwen2.5, phi3, etc.)
- **No API Keys**: Completely local processing with no cloud dependencies
- **Custom Fine-tuning**: Built-in tools for model fine-tuning with your own datasets

### 🧊 3D Model Generation
- **Text-to-3D**: Generate OpenSCAD models from natural language descriptions
- **Real-time Preview**: Live 3D model rendering in the browser
- **Multiple Export Formats**: STL for 3D printing, SCAD for editing, PNG for previews
- **Advanced Prompting**: Sophisticated prompt engineering for better model quality

### 📊 Economic Modeling Tool
- **Project Analysis**: Comprehensive economic metrics calculation (NPV, IRR, payback period)
- **Interactive Dashboard**: Visual representation of economic data
- **Baby Todd Assistant**: AI-powered chat interface for analyzing economic models
- **Model Saving & Loading**: Save and retrieve your economic models for future reference

### 🛠 Multi-Tool Interface
- **Modular Design**: Easy-to-extend architecture for additional AI tools
- **Tool Management**: Centralized configuration and model selection
- **Debug Console**: Comprehensive logging and troubleshooting tools

### 📊 Dataset & Training Tools
- **Dataset Builder**: Interactive tool for creating training datasets
- **Fine-tuning Pipeline**: Complete workflow using Axolotl framework
- **Model Management**: Import and manage custom fine-tuned models

## 🔧 Technology Stack

### Core Libraries
- **Streamlit** (1.22+): Web interface and application framework
- **Ollama**: Local LLM inference engine
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization for economic models
- **Requests**: HTTP client for Ollama API communication

### 3D Processing
- **subprocess**: OpenSCAD process management
- **PIL (Pillow)**: Image processing for model previews
- **base64**: File encoding for downloads

### Data & Training
- **json**: Dataset management and configuration
- **pandas**: Data manipulation for training datasets
- **axolotl**: Fine-tuning framework (external dependency)

### Development Tools
- **pathlib**: Modern file path handling
- **logging**: Comprehensive debug logging
- **os/sys**: System integration and environment management

## 📋 Prerequisites

### Required Software
- **Python 3.8+** with pip
- **Ollama** (latest version) - for AI assistant functionality

### System Requirements
- **RAM**: Minimum 8GB (16GB+ recommended for larger models)
- **Storage**: 10GB+ free space for models and generated files
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)

## 🛠 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/USERNAME/LocalAIAssistant.git
cd LocalAIAssistant
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Install Ollama (Optional for AI Assistant)
```bash
# Download and install Ollama from https://ollama.com/download

# Pull recommended models
ollama pull llama3.2:latest
# Or other models like:
ollama pull qwen2.5:7b
ollama pull phi3:medium

# Verify installation
ollama list
```

## 🚀 Usage Guide

### Starting the Economic Modeling Tool
```bash
# Ensure virtual environment is activated
streamlit run tools/economic_model.py

# Application will open at http://localhost:8501
```

### Economic Model Tool Workflow

#### Basic Usage
1. **Launch Tool**: Run the command above to start the economic modeling tool
2. **Enter Project Data**: Fill in the economic parameters (capital expenditure, revenue, etc.)
3. **Project Details**: Add optional details like project name, description, and location
4. **Calculate Metrics**: Click "Calculate Metrics" to generate economic analysis
5. **View Results**: Examine NPV, IRR, payback period, and other metrics
6. **Save Model**: Your model will be saved automatically for future reference
7. **Chat with Baby Todd**: Ask the assistant questions about your economic analysis

#### Optional Data Upload
- **Spreadsheet Upload**: Upload Excel or CSV files with project data
- **Enhanced Analysis**: The tool will process your data and provide additional insights
- **Download Results**: Export the analysis as a CSV file for further processing

#### Example Economic Analysis Questions
```
"What's a good IRR for this type of project?"
"How does changing the discount rate affect my NPV?"
"What's the break-even point for this project?"
"How does this project compare to industry benchmarks?"
```

## 📁 Project Structure

```
LocalAIAssistant/
├── api/
│   └── __init__.py        # API utilities for Ollama integration
├── tools/
│   ├── economic_model.py  # Economic modeling and analysis tool
│   └── other_tools.py     # Additional tools (future development)
├── data/
│   └── economic_models/   # Saved economic models storage
├── docs/
│   └── images/            # Documentation images
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Optional: Set custom paths
export OLLAMA_HOST="http://localhost:11434"
export MODEL_STORAGE_PATH="./data/economic_models"
```

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Streamlit Installation Issues
```bash
# Make sure you have the latest pip
python -m pip install --upgrade pip

# Try installing streamlit separately
pip install streamlit

# Check installation
streamlit --version
```

#### Ollama Connection Issues (for AI Assistant)
```bash
# Check if Ollama is running
ollama serve

# Verify model availability
ollama list

# Test API connection
curl http://localhost:11434/api/tags
```

#### Memory Issues
- Use smaller models if using the AI assistant feature
- Close other applications to free RAM
- Consider using GPU acceleration if available

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/your-feature-name`
3. **Make changes**: Follow existing code style and patterns
4. **Add tests**: Ensure new features are properly tested
5. **Submit PR**: Provide clear description of changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Ollama](https://ollama.com/)** - Local LLM inference platform
- **[Streamlit](https://streamlit.io/)** - Rapid web app development framework
- **Open Source Community** - For continuous inspiration and contributions

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/patrickcastr/toddler/issues)
- **Discussions**: [GitHub Discussions](https://github.com/patrickcastr/toddler/discussions)

**Made with ❤️ for the local AI community**
