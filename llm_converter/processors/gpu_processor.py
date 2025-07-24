"""GPU processor with OCR capabilities for images and PDFs."""

import os
import json
import logging
import tempfile
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import BaseProcessor
from ..result import ConversionResult
from ..exceptions import ConversionError, FileNotFoundError
from ..pipeline.ocr_service import OCRServiceFactory

# Configure logging
logger = logging.getLogger(__name__)


class GPUConversionResult(ConversionResult):
    """Enhanced ConversionResult for GPU processing with Nanonets OCR capabilities."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, 
                 gpu_processor: Optional['GPUProcessor'] = None, file_path: Optional[str] = None,
                 ocr_provider: str = "nanonets"):
        super().__init__(content, metadata)
        self.gpu_processor = gpu_processor
        self.file_path = file_path
        self.ocr_provider = ocr_provider
        
        # Add GPU-specific metadata
        if metadata is None:
            self.metadata = {}
        
        # Ensure GPU-specific metadata is present
        if 'processing_mode' not in self.metadata:
            self.metadata['processing_mode'] = 'gpu'
        if 'ocr_provider' not in self.metadata:
            self.metadata['ocr_provider'] = ocr_provider
        if 'gpu_processing' not in self.metadata:
            self.metadata['gpu_processing'] = True
    
    def get_ocr_info(self) -> Dict[str, Any]:
        """Get information about the OCR processing used.
        
        Returns:
            Dictionary with OCR processing information
        """
        return {
            'ocr_provider': self.ocr_provider,
            'processing_mode': 'gpu',
            'file_path': self.file_path,
            'gpu_processor_available': self.gpu_processor is not None
        }
    
    def to_markdown(self) -> str:
        """Export as markdown with GPU processing metadata."""
        # Add GPU processing header if not present
        if not self.content.startswith('# GPU Processed Document'):
            header = f"# GPU Processed Document\n\n*Processed using {self.ocr_provider} OCR on GPU*\n\n---\n\n"
            return header + self.content
        return self.content
    
    def to_html(self) -> str:
        """Export as HTML with GPU processing styling."""
        # Get the base HTML from parent class
        html_content = super().to_html()
        
        # Add GPU processing indicator
        gpu_indicator = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; text-align: center;">
            <strong>🚀 GPU Processed</strong> - Enhanced with {self.ocr_provider} OCR
        </div>
        """
        
        # Insert the indicator after the opening body tag
        body_start = html_content.find('<body')
        if body_start != -1:
            body_end = html_content.find('>', body_start) + 1
            return html_content[:body_end] + gpu_indicator + html_content[body_end:]
        
        return html_content
    
    def to_json(self) -> Dict[str, Any]:
        """Export as structured JSON using Nanonets model with specific prompt."""
        print("=== GPUConversionResult.to_json() called ===")
        print(f"gpu_processor: {self.gpu_processor}")
        print(f"file_path: {self.file_path}")
        print(f"file_exists: {os.path.exists(self.file_path) if self.file_path else False}")
        
        try:
            # If we have a GPU processor and file path, use the model to extract JSON
            if self.gpu_processor and self.file_path and os.path.exists(self.file_path):
                logger.info("Using Nanonets model for JSON extraction")
                return self._extract_json_with_model()
            else:
                logger.info("Using fallback JSON conversion")
                # Fallback to base JSON conversion
                return self._convert_to_base_json()
        except Exception as e:
            logger.warning(f"Failed to extract JSON with model: {e}. Using fallback conversion.")
            return self._convert_to_base_json()
    
    def _extract_json_with_model(self) -> Dict[str, Any]:
        """Extract structured JSON using Nanonets model with specific prompt."""
        try:
            from PIL import Image
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
            
            # Get the model from the GPU processor's OCR service
            ocr_service = self.gpu_processor._get_ocr_service()
            
            # Access the model components from the OCR service
            if hasattr(ocr_service, 'processor') and hasattr(ocr_service, 'model') and hasattr(ocr_service, 'tokenizer'):
                model = ocr_service.model
                processor = ocr_service.processor
                tokenizer = ocr_service.tokenizer
            else:
                # Fallback: load model directly
                model_path = "nanonets/Nanonets-OCR-s"
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path, 
                    torch_dtype="auto", 
                    device_map="auto"
                )
                model.eval()
                processor = AutoProcessor.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Define the JSON extraction prompt
            prompt = """Extract all information from the above document and return it as a valid JSON object.

Instructions:
- The output should be a single JSON object.
- Keys should be meaningful field names.
- If multiple similar blocks (like invoice items or line items), return a list of JSON objects under a key.
- Use strings for all values.
- Wrap page numbers using: "page_number": "1"
- Wrap watermarks using: "watermark": "CONFIDENTIAL"
- Use ☐ and ☑ for checkboxes.

Example:
{
  "Name": "John Doe",
  "Invoice Number": "INV-4567",
  "Amount Due": "$123.45",
  "Items": [
    {"Description": "Widget A", "Price": "$20"},
    {"Description": "Widget B", "Price": "$30"}
  ],
  "page_number": "1",
  "watermark": "CONFIDENTIAL"
}"""
            
            # Load the image
            image = Image.open(self.file_path)
            
            # Prepare messages for the model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{self.file_path}"},
                    {"type": "text", "text": prompt},
                ]},
            ]
            
            # Apply chat template and process
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            
            # Generate JSON response
            output_ids = model.generate(**inputs, max_new_tokens=15000, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            
            json_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            print(f"json_text: {json_text}")
            
            # Try to parse the JSON response with improved parsing
            def try_parse_json(text):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Try cleaning and reparsing
                    try:
                        text = re.sub(r"(\w+):", r'"\1":', text)  # wrap keys
                        text = text.replace("'", '"')  # replace single quotes
                        return json.loads(text)
                    except:
                        return {"raw_text": text}
            
            # Parse the JSON
            extracted_data = try_parse_json(json_text)
            
            # Create the result structure
            result = {
                "document": extracted_data,
                "conversion_metadata": self.metadata,
                "format": "gpu_structured_json",
                "gpu_processing_info": {
                    'ocr_provider': self.ocr_provider,
                    'processing_mode': 'gpu',
                    'file_path': self.file_path,
                    'gpu_processor_available': self.gpu_processor is not None,
                    'json_extraction_method': 'nanonets_model'
                }
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Failed to extract JSON with model: {e}")
            raise
    
    def _convert_to_base_json(self) -> Dict[str, Any]:
        """Fallback to base JSON conversion method."""
        # Get the base JSON from parent class
        base_json = super().to_json()
        
        # Add GPU-specific metadata
        base_json['gpu_processing_info'] = {
            'ocr_provider': self.ocr_provider,
            'processing_mode': 'gpu',
            'file_path': self.file_path,
            'gpu_processor_available': self.gpu_processor is not None,
            'json_extraction_method': 'fallback_conversion'
        }
        
        # Update the format to indicate GPU processing
        base_json['format'] = 'gpu_structured_json'
        
        return base_json
    
    def to_text(self) -> str:
        """Export as plain text with GPU processing header."""
        header = f"GPU Processed Document (using {self.ocr_provider} OCR)\n"
        header += "=" * 50 + "\n\n"
        return header + self.content
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and information.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'processing_mode': 'gpu',
            'ocr_provider': self.ocr_provider,
            'file_path': self.file_path,
            'content_length': len(self.content),
            'word_count': len(self.content.split()),
            'line_count': len(self.content.split('\n')),
            'gpu_processor_available': self.gpu_processor is not None
        }
        
        # Add metadata if available
        if self.metadata:
            stats['metadata'] = self.metadata
        
        return stats


class GPUProcessor(BaseProcessor):
    """Processor for image files and PDFs with Nanonets OCR capabilities."""
    
    def __init__(self, preserve_layout: bool = True, include_images: bool = False, ocr_enabled: bool = True, use_markdownify: bool = None, ocr_service=None):
        super().__init__(preserve_layout, include_images, ocr_enabled, use_markdownify)
        self._ocr_service = ocr_service
    
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this processor can handle the file
        """
        if not os.path.exists(file_path):
            return False
        
        # Check file extension - ensure file_path is a string
        file_path_str = str(file_path)
        _, ext = os.path.splitext(file_path_str.lower())
        return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.pdf']
    
    def _get_ocr_service(self):
        """Get OCR service instance."""
        if self._ocr_service is not None:
            return self._ocr_service
        # Use Nanonets OCR service by default
        self._ocr_service = OCRServiceFactory.create_service('nanonets')
        return self._ocr_service
    
    def process(self, file_path: str) -> GPUConversionResult:
        """Process image file or PDF with OCR capabilities.
        
        Args:
            file_path: Path to the image file or PDF
            
        Returns:
            GPUConversionResult with extracted content
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file type
            file_path_str = str(file_path)
            _, ext = os.path.splitext(file_path_str.lower())
            
            if ext == '.pdf':
                logger.info(f"Processing PDF file: {file_path}")
                return self._process_pdf(file_path)
            else:
                logger.info(f"Processing image file: {file_path}")
                return self._process_image(file_path)
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise ConversionError(f"GPU processing failed: {e}")
    
    def _process_image(self, file_path: str) -> GPUConversionResult:
        """Process image file with OCR capabilities.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            GPUConversionResult with extracted content
        """
        # Get OCR service
        ocr_service = self._get_ocr_service()
        
        # Extract text with layout awareness if enabled
        if self.ocr_enabled and self.preserve_layout:
            logger.info("Extracting text with layout awareness using Nanonets OCR")
            extracted_text = ocr_service.extract_text_with_layout(file_path)
        elif self.ocr_enabled:
            logger.info("Extracting text without layout awareness using Nanonets OCR")
            extracted_text = ocr_service.extract_text(file_path)
        else:
            logger.warning("OCR is disabled, returning empty content")
            extracted_text = ""
        
        # Create GPU result
        result = GPUConversionResult(
            content=extracted_text,
            metadata={
                'file_path': file_path,
                'file_type': 'image',
                'ocr_enabled': self.ocr_enabled,
                'preserve_layout': self.preserve_layout,
                'ocr_provider': 'nanonets'
            },
            gpu_processor=self,
            file_path=file_path,
            ocr_provider='nanonets'
        )
        
        logger.info(f"Image processing completed. Extracted {len(extracted_text)} characters")
        return result
    
    def _process_pdf(self, file_path: str) -> GPUConversionResult:
        """Process PDF file by converting to images and using OCR.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            GPUConversionResult with extracted content
        """
        try:
            # Convert PDF to images
            image_paths = self._convert_pdf_to_images(file_path)
            
            if not image_paths:
                logger.warning("No pages could be extracted from PDF")
                return GPUConversionResult(
                    content="",
                    metadata={
                        'file_path': file_path,
                        'file_type': 'pdf',
                        'ocr_enabled': self.ocr_enabled,
                        'preserve_layout': self.preserve_layout,
                        'ocr_provider': 'nanonets',
                        'pages_processed': 0
                    },
                    gpu_processor=self,
                    file_path=file_path,
                    ocr_provider='nanonets'
                )
            
            # Process each page with OCR
            all_texts = []
            ocr_service = self._get_ocr_service()
            
            for i, image_path in enumerate(image_paths):
                logger.info(f"Processing PDF page {i+1}/{len(image_paths)}")
                
                try:
                    if self.ocr_enabled and self.preserve_layout:
                        page_text = ocr_service.extract_text_with_layout(image_path)
                    elif self.ocr_enabled:
                        page_text = ocr_service.extract_text(image_path)
                    else:
                        page_text = ""
                    
                    # Add page header and content if there's text
                    if page_text.strip():
                        # Add page header (markdown style)
                        all_texts.append(f"\n## Page {i+1}\n\n")
                        all_texts.append(page_text)
                        
                        # Add horizontal rule after content (except for last page)
                        if i < len(image_paths) - 1:
                            all_texts.append("\n\n---\n\n")
                    
                except Exception as e:
                    logger.error(f"Failed to process page {i+1}: {e}")
                    # Add error page with markdown formatting
                    all_texts.append(f"\n## Page {i+1}\n\n*Error processing this page: {e}*\n\n")
                    if i < len(image_paths) - 1:
                        all_texts.append("---\n\n")
                
                finally:
                    # Clean up temporary image file
                    try:
                        os.unlink(image_path)
                    except:
                        pass
            
            # Combine all page texts
            combined_text = ''.join(all_texts)
            
            # Create result
            result = GPUConversionResult(
                content=combined_text,
                metadata={
                    'file_path': file_path,
                    'file_type': 'pdf',
                    'ocr_enabled': self.ocr_enabled,
                    'preserve_layout': self.preserve_layout,
                    'ocr_provider': 'nanonets',
                    'pages_processed': len(image_paths)
                },
                gpu_processor=self,
                file_path=file_path,
                ocr_provider='nanonets'
            )
            
            logger.info(f"PDF processing completed. Processed {len(image_paths)} pages, extracted {len(combined_text)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            raise ConversionError(f"PDF processing failed: {e}")
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of paths to temporary image files
        """
        try:
            import fitz  # PyMuPDF
            
            # Open the PDF
            pdf_document = fitz.open(pdf_path)
            image_paths = []
            
            # Create temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Convert each page to image
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    
                    # Set zoom factor for better quality
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                    
                    # Render page to image
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Save image
                    image_path = temp_path / f"page_{page_num+1}.png"
                    pix.save(str(image_path))
                    
                    # Copy to a persistent temporary file
                    persistent_image_path = tempfile.mktemp(suffix='.png')
                    import shutil
                    shutil.copy2(str(image_path), persistent_image_path)
                    
                    image_paths.append(persistent_image_path)
                
                pdf_document.close()
            
            logger.info(f"Converted PDF to {len(image_paths)} images")
            return image_paths
            
        except ImportError:
            logger.error("PyMuPDF (fitz) not available. Please install it: pip install PyMuPDF")
            raise ConversionError("PyMuPDF is required for PDF processing")
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise ConversionError(f"PDF to image conversion failed: {e}")
    
    @staticmethod
    def predownload_ocr_models():
        """Pre-download OCR models by running a dummy prediction."""
        try:
            from llm_converter.pipeline.ocr_service import OCRServiceFactory
            ocr_service = OCRServiceFactory.create_service('nanonets')
            # Create a blank image for testing
            from PIL import Image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img = Image.new('RGB', (100, 100), color='white')
                img.save(tmp.name)
                ocr_service.extract_text_with_layout(tmp.name)
                os.unlink(tmp.name)
            print("Nanonets OCR models pre-downloaded and cached.")
        except Exception as e:
            print(f"Failed to pre-download Nanonets OCR models: {e}") 