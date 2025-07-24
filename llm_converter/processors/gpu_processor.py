"""GPU processor with OCR capabilities for images and PDFs."""

import os
import logging
import tempfile
from typing import Dict, Any, List
from pathlib import Path

from .base import BaseProcessor
from ..result import ConversionResult
from ..exceptions import ConversionError, FileNotFoundError
from ..pipeline.ocr_service import OCRServiceFactory

# Configure logging
logger = logging.getLogger(__name__)


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
    
    def process(self, file_path: str) -> ConversionResult:
        """Process image file or PDF with OCR capabilities.
        
        Args:
            file_path: Path to the image file or PDF
            
        Returns:
            ConversionResult with extracted content
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
    
    def _process_image(self, file_path: str) -> ConversionResult:
        """Process image file with OCR capabilities.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ConversionResult with extracted content
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
        
        # Create result
        result = ConversionResult(
            content=extracted_text,
            metadata={
                'file_path': file_path,
                'file_type': 'image',
                'ocr_enabled': self.ocr_enabled,
                'preserve_layout': self.preserve_layout,
                'ocr_provider': 'nanonets'
            }
        )
        
        logger.info(f"Image processing completed. Extracted {len(extracted_text)} characters")
        return result
    
    def _process_pdf(self, file_path: str) -> ConversionResult:
        """Process PDF file by converting to images and using OCR.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ConversionResult with extracted content
        """
        try:
            # Convert PDF to images
            image_paths = self._convert_pdf_to_images(file_path)
            
            if not image_paths:
                logger.warning("No pages could be extracted from PDF")
                return ConversionResult(
                    content="",
                    metadata={
                        'file_path': file_path,
                        'file_type': 'pdf',
                        'ocr_enabled': self.ocr_enabled,
                        'preserve_layout': self.preserve_layout,
                        'ocr_provider': 'nanonets',
                        'pages_processed': 0
                    }
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
            result = ConversionResult(
                content=combined_text,
                metadata={
                    'file_path': file_path,
                    'file_type': 'pdf',
                    'ocr_enabled': self.ocr_enabled,
                    'preserve_layout': self.preserve_layout,
                    'ocr_provider': 'nanonets',
                    'pages_processed': len(image_paths)
                }
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