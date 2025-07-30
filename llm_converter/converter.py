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
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cpu_preference: bool = False,
        gpu_preference: bool = False
    ):
        """Initialize the file converter.
        
        Args:
            preserve_layout: Whether to preserve document layout
            include_images: Whether to include images in output
            ocr_enabled: Whether to enable OCR for image and PDF processing
            api_key: API key for cloud processing (optional - get from https://app.nanonets.com/#/keys)
            model: Model to use for cloud processing (gemini, openapi) - only for cloud mode
            cpu_preference: Force local CPU-only processing (disables cloud mode)
            gpu_preference: Force local GPU processing (disables cloud mode, requires GPU)
        
        Note:
            - Cloud mode is the default unless cpu_preference or gpu_preference is specified
        """
        self.preserve_layout = preserve_layout
        self.include_images = include_images
        self.api_key = api_key
        self.model = model
        self.cpu_preference = cpu_preference
        self.gpu_preference = gpu_preference
        
        # Determine processing mode
        # Cloud mode is default unless CPU/GPU preference is explicitly set
        self.cloud_mode = not (self.cpu_preference or self.gpu_preference)
        
        # Validate CPU/GPU preferences
        if self.cpu_preference and self.gpu_preference:
            raise ValueError("Cannot specify both cpu_preference and gpu_preference. Choose one or neither.")
        
        # Check GPU availability if GPU preference is set
        if self.gpu_preference and not should_use_gpu_processor():
            raise RuntimeError(
                "GPU preference specified but no GPU is available. "
                "Please ensure CUDA is installed and a compatible GPU is present, "
                "or use cpu_preference=True for CPU-only processing."
            )
        
        # Default to True if not explicitly set
        if ocr_enabled is None:
            self.ocr_enabled = True
        else:
            self.ocr_enabled = ocr_enabled
        
        # Try to get API key from environment if not provided
        if self.cloud_mode and not self.api_key:
            self.api_key = os.environ.get('NANONETS_API_KEY')
        
        # Initialize processors
        self.processors = []
        
        if self.cloud_mode:
            # Cloud mode setup
            cloud_processor = CloudProcessor(
                api_key=self.api_key,  # Can be None for rate-limited access
                model_type=self.model,
                preserve_layout=preserve_layout,
                include_images=include_images
            )
            self.processors.append(cloud_processor)
            
            # if self.api_key:
            #     logger.info("Cloud processing enabled with API key - unlimited access")
            # else:
            #     logger.info("Cloud processing enabled without API key - using rate-limited free tier")
            #     logger.warning("For unlimited access, provide an API key from https://app.nanonets.com/#/keys")
        else:
            # Local mode setup
            logger.info("Local processing mode enabled")
            self._setup_local_processors()
    
    def _setup_local_processors(self):
        """Setup local processors based on CPU/GPU preferences."""
        local_processors = [
            PDFProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images, ocr_enabled=self.ocr_enabled),
            DOCXProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            TXTProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            ExcelProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            HTMLProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            PPTXProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
            ImageProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images, ocr_enabled=self.ocr_enabled),
            URLProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images),
        ]
        
        # Add GPU processor based on preferences and availability
        gpu_available = should_use_gpu_processor()
        
        if self.cpu_preference:
            logger.info("CPU preference specified - using CPU-based processors only")
        elif self.gpu_preference:
            if gpu_available:
                logger.info("GPU preference specified - adding GPU processor with Nanonets OCR")
                gpu_processor = GPUProcessor(preserve_layout=self.preserve_layout, include_images=self.include_images, ocr_enabled=self.ocr_enabled)
                local_processors.append(gpu_processor)
            else:
                # This should not happen due to validation in __init__, but just in case
                raise RuntimeError("GPU preference specified but no GPU is available")
        
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
    
    def get_processing_mode(self) -> str:
        """Get the current processing mode.
        
        Returns:
            String describing the current processing mode
        """
        if self.cloud_mode and self.api_key:
            return "cloud"
        elif self.cpu_preference:
            return "cpu_forced"
        elif self.gpu_preference:
            return "gpu_forced"
        elif should_use_gpu_processor():
            return "gpu_auto"
        else:
            return "cpu_auto"
    
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
        gpu_available = should_use_gpu_processor()
        
        # Try GPU processor only if format is supported AND (gpu_preference OR auto-gpu)
        if not self.cpu_preference and ext in gpu_supported_formats and (self.gpu_preference or (gpu_available and not self.gpu_preference)):
            for processor in self.processors:
                if isinstance(processor, GPUProcessor):
                    if self.gpu_preference:
                        logger.info(f"Using GPU processor with Nanonets OCR for {file_path} (GPU preference specified)")
                    else:
                        logger.info(f"Using GPU processor with Nanonets OCR for {file_path} (GPU available and format supported)")
                    return processor
        
        # Fallback to normal processor selection (CPU processors)
        for processor in self.processors:
            if processor.can_process(file_path):
                # Skip GPU processor in fallback mode to avoid infinite loops
                if isinstance(processor, GPUProcessor):
                    continue
                logger.info(f"Using {processor.__class__.__name__} for {file_path}")
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