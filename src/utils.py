"""
Utility Functions Module

Provides common utility functions including path validation, logging configuration, etc.
"""
import os
import logging
from typing import Tuple, List


def setup_logger(name: str = 'RoutesFormer', level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger
    
    Args:
        name: Logger name
        level: Log level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handler addition
    if not logger.handlers:
        # Console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formatting
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


def is_path_subsequence_of_path(subsequence: Tuple, sequence: List) -> bool:
    """
    Check if subsequence is a subsequence of sequence
    
    Args:
        subsequence: subsequence (discontinuous path)
        sequence: complete sequence (predicted path)
    
    Returns:
        True if subsequence is a subsequence of sequence
    
    Examples:
        >>> is_path_subsequence_of_path((1, 3, 5), [1, 2, 3, 4, 5])
        True
        >>> is_path_subsequence_of_path((1, 4, 2), [1, 2, 3, 4, 5])
        False
    """
    if len(subsequence) == 0:
        return True
    
    sub_idx = 0
    for item in sequence:
        if sub_idx < len(subsequence) and item == subsequence[sub_idx]:
            sub_idx += 1
            if sub_idx == len(subsequence):
                return True
    
    return sub_idx == len(subsequence)


def get_num_links(network) -> int:
    """
    Get number of links in network
    
    Args:
        network: NetworkXGraph object
    
    Returns:
        Number of links
    """
    try:
        if hasattr(network, 'edges'):
            try:
                return len(list(network.edges))
            except Exception:
                return len(network.edges)
        
        if isinstance(network, dict):
            if 'links' in network and isinstance(network['links'], (list, tuple)):
                return len(network['links'])
            if 'segments' in network and isinstance(network['segments'], (list, tuple)):
                return len(network['segments'])
    except Exception:
        pass
    
    return 0


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if not
    
    Args:
        directory: directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time display
    
    Args:
        seconds: seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}hours"

