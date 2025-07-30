"""Cloud processor for Nanonets API integration."""

import os
import requests
import json
import logging
from typing import Dict, Any, Optional

from .base import BaseProcessor
from ..result import ConversionResult
from ..exceptions import ConversionError

logger = logging.getLogger(__name__)


class CloudConversionResult(ConversionResult):
    """Enhanced ConversionResult for cloud mode with lazy API calls."""
    
    def __init__(self, file_path: str, cloud_processor: 'CloudProcessor', metadata: Optional[Dict[str, Any]] = None):
        # Initialize with empty content - we'll make API calls on demand
        super().__init__("", metadata)
        self.file_path = file_path
        self.cloud_processor = cloud_processor
        self._cached_outputs = {}  # Cache API responses by output type
    
    def _get_cloud_output(self, output_type: str, specified_fields: Optional[list] = None, json_schema: Optional[dict] = None) -> str:
        """Get output from cloud API for specific type, with caching."""
        # Validate output type
        valid_output_types = ["markdown", "flat-json", "html", "csv", "specified-fields", "specified-json"]
        if output_type not in valid_output_types:
            logger.warning(f"Invalid output type '{output_type}' for cloud API. Using 'markdown'.")
            output_type = "markdown"
        
        # Create cache key based on output type and parameters
        cache_key = output_type
        if specified_fields:
            cache_key += f"_fields_{','.join(specified_fields)}"
        if json_schema:
            cache_key += f"_schema_{hash(str(json_schema))}"
        
        if cache_key in self._cached_outputs:
            return self._cached_outputs[cache_key]
        
        try:
            # Prepare headers - API key is optional
            headers = {
                'accept': 'application/json'
            }
            if self.cloud_processor.api_key:
                headers['Authorization'] = f'Bearer {self.cloud_processor.api_key}'
            
            # Prepare file for upload
            with open(self.file_path, 'rb') as file:
                files = {
                    'file': (os.path.basename(self.file_path), file, self.cloud_processor._get_content_type(self.file_path))
                }
                
                data = {}
                
                # Add format parameter for JSON output
                if output_type in ["flat-json", "specified-fields", "specified-json"]:
                    data['format'] = 'json'
                # For markdown, html, csv - no format parameter needed (defaults to markdown)
                
                # Add model_type if specified
                if self.cloud_processor.model_type:
                    data['model_type'] = self.cloud_processor.model_type
                
                # Add field extraction parameters
                if output_type == "specified-fields" and specified_fields:
                    data['specified_fields'] = ','.join(specified_fields)
                elif output_type == "specified-json" and json_schema:
                    data['json_schema'] = json.dumps(json_schema)
                
                # Log the request
                if self.cloud_processor.api_key:
                    logger.info(f"Making cloud API call with API key for {output_type} on {self.file_path}")
                else:
                    logger.info(f"Making cloud API call without API key (rate-limited) for {output_type} on {self.file_path}")
                
                # Make API request
                response = requests.post(
                    self.cloud_processor.api_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300
                )
                
                # Handle rate limiting (429) specifically
                if response.status_code == 429:
                    if not self.cloud_processor.api_key:
                        error_msg = (
                            "Rate limit exceeded for free tier usage. "
                            "To get unlimited access, please provide an API key from https://app.nanonets.com/#/keys\n"
                            "Usage: FileConverter(api_key='your_api_key_here')"
                        )
                        logger.error(error_msg)
                        raise ConversionError(error_msg)
                    else:
                        error_msg = "Rate limit exceeded even with API key. Please try again later."
                        logger.error(error_msg)
                        raise ConversionError(error_msg)
                
                response.raise_for_status()
                result_data = response.json()
                
                # Extract content from response
                content = self.cloud_processor._extract_content_from_response(result_data)
                
                # Cache the result
                self._cached_outputs[cache_key] = content
                return content
                
        except ConversionError:
            # Re-raise ConversionError (like rate limiting) without fallback
            raise
        except Exception as e:
            logger.error(f"Failed to get {output_type} from cloud API: {e}")
            # Try fallback to local conversion for other errors
            return self._convert_locally(output_type)
    
    def _convert_locally(self, output_type: str) -> str:
        """Fallback to local conversion methods."""
        if output_type == "html":
            return super().to_html()
        elif output_type == "flat-json":
            return json.dumps(super().to_json(), indent=2)
        elif output_type == "csv":
            return super().to_csv(include_all_tables=True)
        else:
            return self.content
    
    def to_markdown(self) -> str:
        """Export as markdown."""
        return self._get_cloud_output("markdown")
    
    def to_html(self) -> str:
        """Export as HTML."""
        return self._get_cloud_output("html")
    
    def to_json(self, specified_fields: Optional[list] = None, json_schema: Optional[dict] = None) -> Dict[str, Any]:
        """Export as structured JSON with optional field extraction.
        
        Args:
            specified_fields: Optional list of specific fields to extract
            json_schema: Optional JSON schema defining fields and types to extract
            
        Returns:
            Structured JSON with extracted data
        """
        try:
            if specified_fields:
                # Request specified fields extraction
                content = self._get_cloud_output("specified-fields", specified_fields=specified_fields)
                extracted_data = json.loads(content)
                return {
                    "extracted_fields": extracted_data,
                    "requested_fields": specified_fields,
                    "conversion_metadata": self.metadata,
                    "format": "specified_fields"
                }
            
            elif json_schema:
                # Request JSON schema extraction
                content = self._get_cloud_output("specified-json", json_schema=json_schema)
                extracted_data = json.loads(content)
                return {
                    "structured_data": extracted_data,
                    "schema": json_schema,
                    "conversion_metadata": self.metadata,
                    "format": "structured_json"
                }
            
            else:
                # Standard JSON extraction - use json_data field directly
                json_content = self._get_cloud_output("flat-json")
                # The API returns json_data field directly, no need to parse
                return {
                    "document": json_content,
                    "conversion_metadata": self.metadata,
                    "format": "cloud_flat_json"
                }
                
        except Exception as e:
            logger.error(f"Failed to parse JSON content: {e}")
            return {
                "document": {"raw_content": content if 'content' in locals() else ""},
                "conversion_metadata": self.metadata,
                "format": "json_parse_error",
                "error": str(e)
            }
    

    
    def to_text(self) -> str:
        """Export as plain text."""
        # For text output, we can try markdown first and then convert to text
        try:
            return self._get_cloud_output("markdown")
        except Exception as e:
            logger.error(f"Failed to get text output: {e}")
            return ""
    
    def to_csv(self, table_index: int = 0, include_all_tables: bool = False) -> str:
        """Export tables as CSV format.
        
        Args:
            table_index: Which table to export (0-based index). Default is 0 (first table).
            include_all_tables: If True, export all tables with separators. Default is False.
        
        Returns:
            CSV formatted string of the table(s)
        
        Raises:
            ValueError: If no tables are found or table_index is out of range
        """
        return self._get_cloud_output("csv")


class CloudProcessor(BaseProcessor):
    """Processor for cloud-based document conversion using Nanonets API."""
    
    def __init__(self, api_key: Optional[str] = None, output_type: str = None, model_type: Optional[str] = None, 
                 specified_fields: Optional[list] = None, json_schema: Optional[dict] = None, **kwargs):
        """Initialize the cloud processor.
        
        Args:
            api_key: API key for cloud processing (optional - uses rate-limited free tier without key)
            output_type: Output type for cloud processing (markdown, flat-json, html, csv, specified-fields, specified-json)
            model_type: Model type for cloud processing (gemini, openapi)
            specified_fields: List of fields to extract (for specified-fields output type)
            json_schema: JSON schema defining fields and types to extract (for specified-json output type)
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.output_type = output_type
        self.model_type = model_type
        self.specified_fields = specified_fields
        self.json_schema = json_schema
        self.api_url = "https://data-extractor.nanonets.com/ocr"
        
        # Don't validate output_type during initialization - it will be validated during processing
        # This prevents warnings during FileConverter initialization
    
    def can_process(self, file_path: str) -> bool:
        """Check if the processor can handle the file."""
        # Cloud processor supports most common document formats
        # API key is optional - without it, uses rate-limited free tier
        supported_extensions = {
            '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', 
            '.txt', '.html', '.htm', '.png', '.jpg', '.jpeg', '.gif', 
            '.bmp', '.tiff', '.tif'
        }
        
        _, ext = os.path.splitext(file_path.lower())
        return ext in supported_extensions
    
    def process(self, file_path: str) -> CloudConversionResult:
        """Create a lazy CloudConversionResult that will make API calls on demand.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            CloudConversionResult that makes API calls when output methods are called
            
        Raises:
            ConversionError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise ConversionError(f"File not found: {file_path}")
        
        # Create metadata without making any API calls
        metadata = {
            'source_file': file_path,
            'processing_mode': 'cloud',
            'api_provider': 'nanonets',
            'file_size': os.path.getsize(file_path),
            'model_type': self.model_type,
            'has_api_key': bool(self.api_key)
        }
        
        if self.api_key:
            logger.info(f"Created cloud converter for {file_path} with API key - unlimited access")
        else:
            logger.info(f"Created cloud converter for {file_path} without API key - rate-limited access")
        
        # Return lazy result that will make API calls when needed
        return CloudConversionResult(
            file_path=file_path,
            cloud_processor=self,
            metadata=metadata
        )
    
    def _extract_content_from_response(self, response_data: Dict[str, Any]) -> str:
        """Extract content from API response."""
        try:
            # Check for success first
            if not response_data.get('success', True):
                error_msg = response_data.get('message', 'API request failed')
                raise ConversionError(f"API request failed: {error_msg}")
            
            # API returns different fields based on format
            if 'json_data' in response_data and response_data['json_data'] is not None:
                # For JSON requests, return the json_data directly
                return response_data['json_data']
            elif 'markdown' in response_data and response_data['markdown']:
                return response_data['markdown']
            elif 'text' in response_data and response_data['text']:
                return response_data['text']
            elif 'content' in response_data:
                return response_data['content']
            
            # Fallback: return whole response as JSON if no content field
            logger.warning("No content fields found in API response, returning full response")
            return json.dumps(response_data, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to extract content from API response: {e}")
            return json.dumps(response_data, indent=2)
    
    def _get_content_type(self, file_path: str) -> str:
        """Get content type for file upload."""
        _, ext = os.path.splitext(file_path.lower())
        
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        
        return content_types.get(ext, 'application/octet-stream') 