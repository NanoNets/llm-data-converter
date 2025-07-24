"""GPU utility functions for detecting and managing GPU availability."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def is_gpu_available() -> bool:
    """Check if GPU is available for deep learning models.
    
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"GPU detected: {gpu_name} (count: {gpu_count})")
            return True
        else:
            logger.info("No CUDA GPU available")
            return False
    except ImportError:
        logger.info("PyTorch not available, assuming no GPU")
        return False
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
        return False


def get_gpu_info() -> Dict:
    """Get detailed GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "available": False,
        "count": 0,
        "names": [],
        "memory": []
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["count"] = torch.cuda.device_count()
            info["names"] = [torch.cuda.get_device_name(i) for i in range(info["count"])]
            info["memory"] = [torch.cuda.get_device_properties(i).total_memory for i in range(info["count"])]
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
    
    return info


def should_use_gpu_processor() -> bool:
    """Determine if GPU processor should be used based on GPU availability.
    
    Returns:
        True if GPU processor should be used, False otherwise
    """
    return is_gpu_available()


def get_processor_preference() -> str:
    """Get the preferred processor type based on system capabilities.
    
    Returns:
        'gpu' if GPU is available, 'cpu' otherwise
    """
    if should_use_gpu_processor():
        return 'gpu'
    else:
        return 'cpu' 