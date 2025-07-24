"""Image file processor with OCR capabilities."""

import os
import logging
from typing import Dict, Any

from .base import BaseProcessor
from ..result import ConversionResult
from ..exceptions import ConversionError, FileNotFoundError
from ..pipeline.ocr_service import OCRServiceFactory

# Configure logging
logger = logging.getLogger(__name__)


class GPUProcessor(BaseProcessor):
    """Processor for image files (JPG, PNG, etc.) with OCR capabilities."""
    
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
        return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
    
    def _get_ocr_service(self):
        """Get OCR service instance."""
        if self._ocr_service is not None:
            return self._ocr_service
        # Use Nanonets OCR service by default
        self._ocr_service = OCRServiceFactory.create_service('nanonets')
        return self._ocr_service
    
    def process(self, file_path: str) -> ConversionResult:
        """Process image file with OCR capabilities.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ConversionResult with extracted content
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            logger.info(f"Processing image file: {file_path}")
            
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
            
        except Exception as e:
            logger.error(f"Failed to process image file {file_path}: {e}")
            raise ConversionError(f"Image processing failed: {e}")
    
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