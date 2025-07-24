# LLM Data Converter

[![PyPI version](https://badge.fury.io/py/llm-data-converter.svg?v=2)](https://badge.fury.io/py/llm-data-converter)
[![GitHub stars](https://img.shields.io/github/stars/NanoNets/llm-data-converter?style=social)](https://github.com/NanoNets/llm-data-converter)
[![Downloads](https://pepy.tech/badge/llm-data-converter)](https://pepy.tech/project/llm-data-converter)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-data-converter)](https://pypi.org/project/llm-data-converter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **🆓 Try Cloud Mode for Free:** Test the cloud extraction capabilities at [https://extraction-api.nanonets.com/](https://extraction-api.nanonets.com/) - API key required for the web interface!

Convert any document format into LLM-ready data format (markdown) with advanced intelligent document processing capabilities powered by pre-trained models.

**🆕 NEW: Cloud Mode Available!** - Process documents using the powerful Nanonets cloud API with a free API key for faster, more accurate results.

## Installation

```bash
pip install llm-data-converter
```

**Requirements:**
- Python 3.8 or higher

### System Dependencies for Intelligent Document Processing

For this library to work properly, you may need to install additional system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 libgomp1
pip install setuptools
```

**macOS:**
```bash
# Usually not needed, but if you encounter OpenGL issues:
brew install mesa
```

**Note:** The package will automatically download and cache intelligent models on first use. For cloud mode, no system dependencies or model downloads are required.

## Quick Start

```python
from llm_converter import FileConverter

# Local mode (default) - works offline
converter = FileConverter()
result = converter.convert("document.pdf").to_markdown()
print(result)
```

**Cloud Mode (New!)** - For faster, more accurate results:
```python
from llm_converter import FileConverter

# Only api_key is required for cloud mode
# Get API key from https://app.nanonets.com/#/keys
converter = FileConverter(cloud_mode=True, api_key="your_api_key")
result = converter.convert("document.pdf").to_markdown()  # Same interface!
print(result)

# Optional: Choose specific model for cloud processing
converter = FileConverter(cloud_mode=True, api_key="your_api_key", model="gemini")  # model is optional
result = converter.convert("document.pdf").to_markdown()
print(result)
```

**CPU/GPU Processing Preferences:**
```python
from llm_converter import FileConverter

# Force CPU-only processing (useful for servers without GPU)
converter = FileConverter(cpu_preference=True)
result = converter.convert("document.pdf").to_markdown()

# Force GPU processing (will raise error if GPU not available)
converter = FileConverter(gpu_preference=True)
result = converter.convert("document.pdf").to_markdown()

# Auto-detect (default) - uses GPU if available, falls back to CPU
converter = FileConverter()  # No preference specified
result = converter.convert("document.pdf").to_markdown()

# Check current processing mode
print(f"Processing mode: {converter.get_processing_mode()}")  # cpu_forced, gpu_forced, gpu_auto, cpu_auto, or cloud
```

## Features

- **Multiple Input Formats**: PDF, DOCX, TXT, HTML, URLs, Excel files, and more
- **Multiple Output Formats**: Markdown, HTML, JSON, Plain Text
- **LLM Integration**: Seamless integration with LiteLLM and other LLM libraries
- **Local Processing**: Process documents locally without external dependencies
- **Cloud Processing**: Fast, accurate processing with Nanonets cloud API
- **Layout Preservation**: Maintain document structure and formatting
- **CPU/GPU Processing Modes**: Flexible processing options:
  - **Auto-detect**: Automatically uses GPU if available, falls back to CPU
  - **CPU-only**: Force CPU processing (useful for servers without GPU)
  - **GPU-only**: Force GPU processing (requires CUDA-compatible GPU)
- **Intelligent Document Processing**: Advanced document understanding and conversion powered by pre-trained models:
  - **Layout Detection**: Intelligent models for document structure understanding
  - **Text Recognition**: High-accuracy text extraction with confidence scoring
  - **Table Structure**: Intelligent table detection and conversion to markdown format
  - **Automatic Model Download**: Models are automatically downloaded and cached

## Usage Examples

### Convert PDF to Markdown

```python
from llm_converter import FileConverter

# Local mode (default)
converter = FileConverter()
result = converter.convert("document.pdf").to_markdown()
print(result)

# Cloud mode (just add cloud_mode=True and api_key)
converter = FileConverter(cloud_mode=True, api_key="your_api_key")
result = converter.convert("document.pdf").to_markdown()  # Same interface!
print(result)
```

### Convert Image to HTML

```python
from llm_converter import FileConverter

converter = FileConverter()
result = converter.convert("sample.png").to_html()
print(result)
```

### Chain with LLM

```python
from llm_converter import FileConverter
from litellm import completion

converter = FileConverter()
document_content = converter.convert("report.pdf").to_markdown()

# Use with any LLM
response = completion(
    model="openai/gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that analyzes documents."},
        {"role": "user", "content": f"Summarize this document:\n\n{document_content}"}
    ]
)

print(response.choices[0].message.content)
```

## Supported Formats

### Input Formats
- **Documents**: PDF, DOCX, TXT
- **Web**: URLs, HTML files
- **Data**: Excel (XLSX, XLS), CSV
- **Images**: PNG, JPG, JPEG 

### Output Formats
- **Markdown**: Clean, structured markdown with proper table formatting
- **HTML**: Formatted HTML with styling
- **JSON**: Structured JSON data
- **Plain Text**: Simple text extraction

## CLI usage

The `llm-converter` command-line tool provides easy access to all conversion features:

### Basic Usage

```bash
# Convert a PDF to markdown (default)
llm-converter document.pdf

# Convert to different output formats
llm-converter document.pdf --output html
llm-converter document.pdf --output json
llm-converter document.pdf --output text
```

### Cloud Mode

```bash
# Convert using cloud API - only api_key required
llm-converter document.pdf --cloud-mode --api-key YOUR_API_KEY

# Use environment variable for API key
export NANONETS_API_KEY=your_api_key
llm-converter document.pdf --cloud-mode --output json

# Optional: Use specific model for cloud processing
llm-converter document.pdf --cloud-mode --api-key YOUR_KEY --model gemini
llm-converter document.pdf --cloud-mode --model openapi --output json
```

### Advanced Options

```bash
# Save output to file
llm-converter document.pdf --output-file output.md

# Force CPU-only processing
llm-converter document.pdf --cpu-only

# Force GPU processing (will error if GPU not available)
llm-converter document.pdf --gpu-only

# For image input
llm-converter image.png 

# Convert multiple files at once
llm-converter file1.pdf file2.docx file3.xlsx --output markdown
```

### List Supported Formats

```bash
# See all supported input formats
llm-converter --list-formats
```

### Examples

```bash
# Convert PDF to markdown
llm-converter scanned_document.pdf --output markdown

# Convert image to HTML with layout preservation
llm-converter screenshot.png --output html

# Convert multiple documents to JSON
llm-converter report.pdf presentation.pptx data.xlsx --output json --output-file combined.json

# Convert URL content to markdown
llm-converter https://blog.example.com --output markdown --output-file blog_content.md

# Cloud mode examples
llm-converter document.pdf --cloud-mode --api-key YOUR_KEY
llm-converter document.pdf --cloud-mode --output json  # env var NANONETS_API_KEY
```

## API Reference for library

### FileConverter

Main class for converting documents to LLM-ready formats.

#### Constructor Parameters

- `preserve_layout: bool = True`: Whether to preserve document layout
- `include_images: bool = True`: Whether to include images in output
- `ocr_enabled: bool = True`: Whether to enable OCR for image and PDF processing
- `cloud_mode: bool = False`: Whether to use cloud processing via Nanonets API
- `api_key: Optional[str] = None`: API key for cloud mode (get from https://app.nanonets.com/#/keys)
- `model: Optional[str] = None`: Model to use for cloud processing (gemini, openapi) - only for cloud mode
- `cpu_preference: bool = False`: Force CPU-only processing (overrides GPU preference)
- `gpu_preference: bool = False`: Force GPU processing (will raise error if GPU not available)

#### Methods

- `convert(file_path: str) -> ConversionResult`: Convert a file to internal format
- `convert_url(url: str) -> ConversionResult`: Convert a URL page contents to internal format
- `convert_text(text: str) -> ConversionResult`: Convert plain text to internal format
- `get_processing_mode() -> str`: Get current processing mode (cpu_forced, gpu_forced, gpu_auto, cpu_auto, cloud)

### CloudFileConverter

Extended FileConverter with cloud processing capabilities.

#### Methods

- `convert(file_path: str) -> ConversionResult`: Convert using cloud API (same interface!)
- `is_cloud_enabled() -> bool`: Check if cloud processing is available

### ConversionResult

Result object with methods to export to different formats.

#### Methods

- `to_markdown() -> str`: Export as markdown
- `to_html() -> str`: Export as HTML  
- `to_json() -> dict`: Export as JSON
- `to_text() -> str`: Export as plain text

## Troubleshooting

### Cloud Mode Setup

1. Get your free API key from [https://app.nanonets.com/#/keys](https://app.nanonets.com/#/keys)
2. Set environment variable: `export NANONETS_API_KEY=your_key`
3. Or provide directly: `CloudFileConverter(api_key="your_key")`

### Installation Issues

#### Tokenizers Build Error

If you encounter an error like this during installation:
```
ERROR: Could not find a version that satisfies the requirement puccinialin
ERROR: No matching distribution found for puccinialin
```

This is typically caused by the `tokenizers` package failing to build from source. Here are several solutions:

**Solution 1: Update pip and install pre-compiled wheels**
```bash
pip install --upgrade pip
pip install llm-data-converter --no-cache-dir
```

**Solution 2: Install with specific tokenizers version**
```bash
pip install tokenizers==0.21.0
pip install llm-data-converter
```

**Solution 3: Use conda (recommended for complex dependencies)**
```bash
conda install -c conda-forge llm-data-converter
```

#### Numpy/Homebrew Conflict (macOS)

If you see this error on macOS:
```
error: uninstall-no-record-file
× Cannot uninstall numpy 2.1.2
```

**Solution: Use virtual environment (recommended)**
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install llm-data-converter
```

### Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/nanonets/llm-data-converter/issues)
- **Documentation**: Check this README and the [scripts documentation](scripts/README.md)
- **Community**: Join discussions on GitHub

## License

MIT License - see LICENSE file for details. 