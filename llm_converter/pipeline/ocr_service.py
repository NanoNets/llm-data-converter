"""OCR Service abstraction for neural document processing."""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OCRService(ABC):
    """Abstract base class for OCR services."""
    
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        pass
    
    @abstractmethod
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Layout-aware extracted text as markdown
        """
        pass


class NanonetsOCRService(OCRService):
    """Nanonets OCR implementation using NanonetsDocumentProcessor."""
    
    def __init__(self):
        """Initialize the service."""
        from .nanonets_processor import NanonetsDocumentProcessor
        self._processor = NanonetsDocumentProcessor()
        logger.info("NanonetsOCRService initialized")
    
    @property
    def model(self):
        """Get the Nanonets model."""
        return self._processor.model
    
    @property
    def processor(self):
        """Get the Nanonets processor."""
        return self._processor.processor
    
    @property
    def tokenizer(self):
        """Get the Nanonets tokenizer."""
        return self._processor.tokenizer
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Nanonets OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text(image_path)
                logger.info(f"Extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Nanonets OCR extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Nanonets OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text_with_layout(image_path)
                logger.info(f"Layout-aware extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Nanonets OCR layout-aware extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Nanonets OCR layout-aware extraction failed: {e}")
            return ""


class NeuralOCRService(OCRService):
    """Neural OCR implementation using docling's pre-trained models."""
    
    def __init__(self):
        """Initialize the service."""
        from .neural_document_processor import NeuralDocumentProcessor
        self._processor = NeuralDocumentProcessor()
        logger.info("NeuralOCRService initialized")
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Neural OCR (docling models)."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text(image_path)
                logger.info(f"Extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Neural OCR extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Neural OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Neural OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text_with_layout(image_path)
                logger.info(f"Layout-aware extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Neural OCR layout-aware extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Neural OCR layout-aware extraction failed: {e}")
            return ""


class OCRServiceFactory:
    """Factory for creating OCR services based on configuration."""
    
    @staticmethod
    def create_service(provider: str = None) -> OCRService:
        """Create OCR service based on provider configuration.
        
        Args:
            provider: OCR provider name (defaults to config)
            
        Returns:
            OCRService instance
        """
        from llm_converter.config import InternalConfig
        
        if provider is None:
            provider = getattr(InternalConfig, 'ocr_provider', 'nanonets')
        
        if provider.lower() == 'nanonets':
            return NanonetsOCRService()
        elif provider.lower() == 'neural':
            return NeuralOCRService()
        else:
            raise ValueError(f"Unsupported OCR provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available OCR providers.
        
        Returns:
            List of available provider names
        """
        return ['nanonets', 'neural'] 