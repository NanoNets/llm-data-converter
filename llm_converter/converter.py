"""Main converter class for handling document conversion."""

import os
import logging
from typing import List, Optional

from .processors import (
    PDFProcessor,
    DOCXProcessor,
    TXTProcessor,
    ExcelProcessor,
    URLProcessor,
    HTMLProcessor,
    PPTXProcessor,
    ImageProcessor,
    CloudProcessor,
    GPUProcessor,
)
from .result import ConversionResult
from .exceptions import ConversionError, UnsupportedFormatError, FileNotFoundError
from .utils.gpu_utils import should_use_gpu_processor

# Configure logging
logger = logging.getLogger(__name__)


class FileConverter:
    """Main class for converting documents to LLM-ready formats."""
    
    def __init__(
        self,
        preserve_layout: bool = True,
        include_images: bool = True,
        ocr_enabled: bool = True,
        cloud_mode: bool = False,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize the file converter.
        
        Args:
            preserve_layout: Whether to preserve document layout
            include_images: Whether to include images in output
            ocr_enabled: Whether to enable OCR for image and PDF processing
            cloud_mode: Whether to use cloud processing via Nanonets API
            api_key: API key for cloud mode (get from https://app.nanonets.com/#/keys)
            model: Model to use for cloud processing (gemini, openapi) - only for cloud mode
        """
        self.preserve_layout = preserve_layout
        self.include_images = include_images
        self.cloud_mode = cloud_mode
        self.api_key = api_key
        self.model = model
        
        # Default to True if not explicitly set
        if ocr_enabled is None:
            self.ocr_enabled = True
        else:
            self.ocr_enabled = ocr_enabled
        
        # Try to get API key from environment if cloud mode is enabled
        if self.cloud_mode and not self.api_key:
            self.api_key = os.environ.get('NANONETS_API_KEY')
            if not self.api_key:
                logger.warning(
                    "Cloud mode enabled but no API key provided. "
                    "Please provide api_key parameter or set NANONETS_API_KEY environment variable. "
                    "Get your API key from https://app.nanonets.com/#/keys"
                )
        
        # Initialize processors in order of preference
        self.processors = []
        
        # Add cloud processor first if cloud mode is enabled and API key is available
        if self.cloud_mode and self.api_key:
            cloud_processor = CloudProcessor(
                api_key=self.api_key,
                output_type="markdown",  # Default output type
                model_type=self.model,   # Pass model as model_type to cloud processor
                preserve_layout=preserve_layout,
                include_images=include_images
            )
            self.processors.append(cloud_processor)
            logger.info("Cloud processing enabled with Nanonets API - skipping local model initialization")
        else:
            # Only initialize local processors if not in cloud mode or if cloud mode is not properly configured
            if self.cloud_mode and not self.api_key:
                logger.warning("Cloud mode requested but no API key available - falling back to local processing")
            
            logger.info("Initializing local processors...")
            local_processors = [
                PDFProcessor(preserve_layout=preserve_layout, include_images=include_images, ocr_enabled=self.ocr_enabled),
                DOCXProcessor(preserve_layout=preserve_layout, include_images=include_images),
                TXTProcessor(preserve_layout=preserve_layout, include_images=include_images),
                ExcelProcessor(preserve_layout=preserve_layout, include_images=include_images),
                HTMLProcessor(preserve_layout=preserve_layout, include_images=include_images),
                PPTXProcessor(preserve_layout=preserve_layout, include_images=include_images),
                ImageProcessor(preserve_layout=preserve_layout, include_images=include_images, ocr_enabled=self.ocr_enabled),
                URLProcessor(preserve_layout=preserve_layout, include_images=include_images),
            ]
            
            # Add GPU processor only if GPU is available
            if should_use_gpu_processor():
                logger.info("GPU detected - adding GPU processor with Nanonets OCR (supports: .jpg, .jpeg, .png, .bmp, .tiff, .webp, .gif, .pdf)")
                gpu_processor = GPUProcessor(preserve_layout=preserve_layout, include_images=include_images, ocr_enabled=self.ocr_enabled)
                local_processors.append(gpu_processor)
            else:
                logger.info("No GPU detected - using CPU-based processors only")
            
            self.processors.extend(local_processors)
    
    def convert(self, file_path: str) -> ConversionResult:
        """Convert a file to internal format.
        
        Args:
            file_path: Path to the file to convert
            
        Returns:
            ConversionResult containing the processed content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFormatError: If the format is not supported
            ConversionError: If conversion fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Find the appropriate processor
        processor = self._get_processor(file_path)
        if not processor:
            raise UnsupportedFormatError(f"No processor found for file: {file_path}")
        
        logger.info(f"Using processor {processor.__class__.__name__} for {file_path}")
        
        # Process the file
        return processor.process(file_path)
    
    def convert_with_output_type(self, file_path: str, output_type: str) -> ConversionResult:
        """Convert a file with specific output type for cloud processing.
        
        Args:
            file_path: Path to the file to convert
            output_type: Desired output type (markdown, flat-json, html)
            
        Returns:
            ConversionResult containing the processed content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFormatError: If the format is not supported
            ConversionError: If conversion fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # For cloud mode, create a processor with the specific output type
        if self.cloud_mode and self.api_key:
            cloud_processor = CloudProcessor(
                api_key=self.api_key,
                output_type=output_type,
                model_type=self.model,   # Pass model as model_type
                preserve_layout=self.preserve_layout,
                include_images=self.include_images
            )
            if cloud_processor.can_process(file_path):
                logger.info(f"Using cloud processor with output_type={output_type} for {file_path}")
                return cloud_processor.process(file_path)
        
        # Fallback to regular conversion for local mode
        return self.convert(file_path)
    
    def convert_url(self, url: str) -> ConversionResult:
        """Convert a URL to internal format.
        
        Args:
            url: URL to convert
            
        Returns:
            ConversionResult containing the processed content
            
        Raises:
            ConversionError: If conversion fails
        """
        # Cloud mode doesn't support URL conversion
        if self.cloud_mode:
            raise ConversionError("URL conversion is not supported in cloud mode. Use local mode for URL processing.")
        
        # Find the URL processor
        url_processor = None
        for processor in self.processors:
            if isinstance(processor, URLProcessor):
                url_processor = processor
                break
        
        if not url_processor:
            raise ConversionError("URL processor not available")
        
        logger.info(f"Converting URL: {url}")
        return url_processor.process(url)
    
    def convert_text(self, text: str) -> ConversionResult:
        """Convert plain text to internal format.
        
        Args:
            text: Plain text to convert
            
        Returns:
            ConversionResult containing the processed content
        """
        # Cloud mode doesn't support text conversion
        if self.cloud_mode:
            raise ConversionError("Text conversion is not supported in cloud mode. Use local mode for text processing.")
        
        metadata = {
            "content_type": "text",
            "processor": "TextConverter",
            "preserve_layout": self.preserve_layout
        }
        
        return ConversionResult(text, metadata)
    
    def is_cloud_enabled(self) -> bool:
        """Check if cloud processing is enabled and configured.
        
        Returns:
            True if cloud processing is available
        """
        return self.cloud_mode and bool(self.api_key)
    
    def _get_processor(self, file_path: str):
        """Get the appropriate processor for the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processor that can handle the file, or None if none found
        """
        # Define GPU-supported formats
        gpu_supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.pdf']
        
        # Check file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Check if GPU processor should be used for this file type
        if should_use_gpu_processor() and ext in gpu_supported_formats:
            for processor in self.processors:
                if isinstance(processor, GPUProcessor):
                    logger.info(f"Using GPU processor with Nanonets OCR for {file_path} (GPU available and format supported)")
                    return processor
        
        # Fallback to normal processor selection
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        formats = []
        for processor in self.processors:
            if hasattr(processor, 'can_process'):
                # This is a simplified way to get formats
                # In a real implementation, you might want to store this info
                if isinstance(processor, PDFProcessor):
                    formats.extend(['.pdf'])
                elif isinstance(processor, DOCXProcessor):
                    formats.extend(['.docx', '.doc'])
                elif isinstance(processor, TXTProcessor):
                    formats.extend(['.txt', '.text'])
                elif isinstance(processor, ExcelProcessor):
                    formats.extend(['.xlsx', '.xls', '.csv'])
                elif isinstance(processor, HTMLProcessor):
                    formats.extend(['.html', '.htm'])
                elif isinstance(processor, PPTXProcessor):
                    formats.extend(['.ppt', '.pptx'])
                elif isinstance(processor, ImageProcessor):
                    formats.extend(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'])
                elif isinstance(processor, URLProcessor):
                    formats.append('URLs')
                elif isinstance(processor, CloudProcessor):
                    # Cloud processor supports many formats, but we don't want duplicates
                    pass
                elif isinstance(processor, GPUProcessor):
                    # GPU processor supports all image formats and PDFs
                    formats.extend(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.pdf'])
        
        return list(set(formats))  # Remove duplicates 