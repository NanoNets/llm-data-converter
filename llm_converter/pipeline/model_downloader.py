"""Model downloader utility for downloading pre-trained models from Hugging Face."""

import logging
import os
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
from ..utils.gpu_utils import is_gpu_available, get_gpu_info

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Downloads pre-trained models from Hugging Face or Nanonets S3."""
    
    # Nanonets S3 model URLs (primary source)
    S3_BASE_URL = "https://public-vlms.s3-us-west-2.amazonaws.com/llm-data-converter"
    
    # Model configurations with both S3 and HuggingFace sources
    LAYOUT_MODEL = {
        "s3_url": f"{S3_BASE_URL}/layout-model-v2.2.0.tar.gz",
        "repo_id": "ds4sd/docling-models",
        "revision": "v2.2.0",
        "model_path": "model_artifacts/layout",
        "cache_folder": "layout"
    }
    
    TABLE_MODEL = {
        "s3_url": f"{S3_BASE_URL}/tableformer-model-v2.2.0.tar.gz",
        "repo_id": "ds4sd/docling-models", 
        "revision": "v2.2.0",
        "model_path": "model_artifacts/tableformer",
        "cache_folder": "tableformer"
    }
    
    # Nanonets OCR model configuration
    NANONETS_OCR_MODEL = {
        "s3_url": f"{S3_BASE_URL}/Nanonets-OCR-s.tar.gz",
        "repo_id": "nanonets/Nanonets-OCR-s",
        "revision": "main",
        "cache_folder": "nanonets-ocr",
    }
    
    # Note: EasyOCR downloads its own models automatically, no need for custom model
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the model downloader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "llm_converter" / "models"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def download_models(self, force: bool = False, progress: bool = True) -> Path:
        """Download all required models.
        
        Args:
            force: Force re-download even if models exist
            progress: Show download progress
            
        Returns:
            Path to the models directory
        """
        logger.info("Downloading pre-trained models...")
        
        # Auto-detect GPU for Nanonets model
        gpu_available = is_gpu_available()
        print("gpu_available", gpu_available)
        if gpu_available:
            logger.info("GPU detected - including Nanonets OCR model")
        else:
            logger.info("No GPU detected - skipping Nanonets OCR model (CPU-only mode)")
        
        models_to_download = [
            ("Layout Model", self.LAYOUT_MODEL),
            ("Table Structure Model", self.TABLE_MODEL)
        ]
        
        # Add Nanonets OCR model only if GPU is available
        if gpu_available:
            models_to_download.append(("Nanonets OCR Model", self.NANONETS_OCR_MODEL))
        
        for model_name, model_config in models_to_download:
            logger.info(f"Downloading {model_name}...")
            self._download_model(model_config, force, progress)
        
        logger.info("All models downloaded successfully!")
        return self.cache_dir
    
    def _download_model(self, model_config: dict, force: bool, progress: bool):
        """Download a specific model.
        
        Args:
            model_config: Model configuration dictionary
            force: Force re-download
            progress: Show progress
        """
        model_dir = self.cache_dir / model_config["cache_folder"]
        
        if model_dir.exists() and not force:
            logger.info(f"Model already exists at {model_dir}")
            return
        
        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        success = False
        
        # Check if user prefers Hugging Face via environment variable
        prefer_hf = os.environ.get("LLM_CONVERTER_PREFER_HF", "false").lower() == "true"
        
        # Try S3 first (Nanonets hosted models) unless user prefers HF
        if not prefer_hf and "s3_url" in model_config:
            try:
                logger.info(f"Downloading from Nanonets S3: {model_config['s3_url']}")
                self._download_from_s3(
                    s3_url=model_config["s3_url"],
                    local_dir=model_dir,
                    force=force,
                    progress=progress
                )
                success = True
                logger.info("Successfully downloaded from Nanonets S3")
            except Exception as e:
                logger.warning(f"S3 download failed: {e}")
                logger.info("Falling back to Hugging Face...")
        
        # Fallback to Hugging Face if S3 fails
        if not success:
            self._download_from_hf(
                repo_id=model_config["repo_id"],
                revision=model_config["revision"],
                local_dir=model_dir,
                force=force,
                progress=progress
            )
    
    def _download_from_hf(self, repo_id: str, revision: str, local_dir: Path, 
                          force: bool, progress: bool):
        """Download model from Hugging Face using docling's logic.
        
        Args:
            repo_id: Hugging Face repository ID
            revision: Git revision/tag
            local_dir: Local directory to save model
            force: Force re-download
            progress: Show progress
        """
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import disable_progress_bars
            import huggingface_hub
            
            if not progress:
                disable_progress_bars()
            
            # Check if models are already downloaded
            if local_dir.exists() and any(local_dir.iterdir()):
                logger.info(f"Model {repo_id} already exists at {local_dir}")
                return
            
            # Try to download with current authentication
            try:
                download_path = snapshot_download(
                    repo_id=repo_id,
                    force_download=force,
                    local_dir=str(local_dir),
                    revision=revision,
                    token=None,  # Use default token if available
                )
                logger.info(f"Successfully downloaded {repo_id} to {download_path}")
                
            except huggingface_hub.errors.HfHubHTTPError as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.warning(
                        f"Authentication failed for {repo_id}. This model may require a Hugging Face token.\n"
                        "To fix this:\n"
                        "1. Create a free account at https://huggingface.co/\n"
                        "2. Generate a token at https://huggingface.co/settings/tokens\n"
                        "3. Set it as environment variable: export HF_TOKEN='your_token_here'\n"
                        "4. Or run: huggingface-cli login\n\n"
                        "The library will continue with basic OCR capabilities."
                    )
                    # Don't raise the error, just log it and continue
                    return
                else:
                    raise
            
        except ImportError:
            logger.error("huggingface_hub not available. Please install it: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"Failed to download model {repo_id}: {e}")
            # Don't raise for authentication errors - allow fallback processing
            if "401" not in str(e) and "Unauthorized" not in str(e):
                raise

    def get_model_path(self, model_type: str) -> Optional[Path]:
        """Get the path to a specific model.
        
        Args:
            model_type: Type of model ('layout', 'table', 'nanonets-ocr')
            
        Returns:
            Path to the model directory, or None if not found
        """
        model_mapping = {
            'layout': self.LAYOUT_MODEL["cache_folder"],
            'table': self.TABLE_MODEL["cache_folder"],
            'nanonets-ocr': self.NANONETS_OCR_MODEL["cache_folder"]
        }
        
        if model_type not in model_mapping:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        model_path = self.cache_dir / model_mapping[model_type]
        
        if not model_path.exists():
            logger.warning(f"Model {model_type} not found at {model_path}")
            return None
        
        return model_path 

    def are_models_cached(self) -> bool:
        """Check if all required models are cached.
        
        Returns:
            True if all required models are cached, False otherwise
        """
        layout_path = self.get_model_path('layout')
        table_path = self.get_model_path('table')
        
        # Only check for Nanonets model if GPU is available
        if is_gpu_available():
            nanonets_path = self.get_model_path('nanonets-ocr')
            return layout_path is not None and table_path is not None and nanonets_path is not None
        else:
            return layout_path is not None and table_path is not None
    
    def _download_from_s3(self, s3_url: str, local_dir: Path, force: bool, progress: bool):
        """Download model from Nanonets S3.
        
        Args:
            s3_url: S3 URL of the model archive
            local_dir: Local directory to extract model
            force: Force re-download
            progress: Show progress
        """
        import tarfile
        import tempfile
        
        # Download the tar.gz file
        response = requests.get(s3_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            if progress and total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
            
            tmp_file_path = tmp_file.name
        
        try:
            # Extract the archive
            logger.info(f"Extracting model to {local_dir}")
            with tarfile.open(tmp_file_path, 'r:gz') as tar:
                tar.extractall(path=local_dir)
            
            logger.info("Model extraction completed successfully")
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    
    def get_cache_info(self) -> dict:
        """Get information about cached models.
        
        Returns:
            Dictionary with cache information
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'gpu_info': get_gpu_info(),
            'models': {}
        }
        
        # Always check layout and table models
        for model_type in ['layout', 'table']:
            path = self.get_model_path(model_type)
            info['models'][model_type] = {
                'cached': path is not None,
                'path': str(path) if path else None
            }
        
        # Only check Nanonets model if GPU is available
        if is_gpu_available():
            path = self.get_model_path('nanonets-ocr')
            info['models']['nanonets-ocr'] = {
                'cached': path is not None,
                'path': str(path) if path else None,
                'gpu_required': True
            }
        else:
            info['models']['nanonets-ocr'] = {
                'cached': False,
                'path': None,
                'gpu_required': True,
                'skipped': 'No GPU available'
            }
        
        return info 