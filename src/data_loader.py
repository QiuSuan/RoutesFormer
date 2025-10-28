"""
Data Loading Module

Provides training and testing data preparation functions, including:
- Discontinuous path generation
- Training sample preparation
- Sparse observation data simulation (AVI detectors)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def prepare_discontinuous_path(
    path: np.ndarray,
    mask_ratio: Union[float, str]
) -> np.ndarray:
    """
    Generate discontinuous path from complete path (for masked learning during training)
    
    Args:
        path: Complete path, shape (path_length,)
        mask_ratio: Mask ratio, float value indicates the proportion of links to keep, 'OD' means only keep origin and destination
    
    Returns:
        Discontinuous path containing only kept link IDs
    
    Examples:
        >>> path = np.array([1, 2, 3, 4, 5])
        >>> discontinuous_path = prepare_discontinuous_path(path, 0.5)
        # May return [1, 3, 5] (keeping origin, destination and one intermediate point)
    """
    if mask_ratio == 'OD':
        return np.array([path[0], path[-1]])
    
    # Calculate number of links to keep (always preserve origin and destination)
    link_num = max(0, int((1 - mask_ratio) * len(path)) - 2)
    
    # Randomly select intermediate links to keep
    per = np.random.permutation(len(path))[:link_num]
    
    discontinuous_path = []
    for i in range(len(path)):
        if (i == 0 or i == len(path) - 1) or i in per:
            discontinuous_path.append(path[i])
    
    return np.array(discontinuous_path)


def prepare_training_samples(
    gt_dict: Dict,
    mask_ratio: Union[float, str] = 0.15,
    path_idxs: Optional[List[int]] = None
) -> Dict:
    """
    Prepare masked language model (MLM) samples for training
    
    Args:
        gt_dict: Ground truth path dictionary containing 'paths' key
        mask_ratio: Mask ratio
        path_idxs: List of path indices to process, None means process all paths
    
    Returns:
        Dictionary containing discontinuous paths in the format:
        {
            'paths': {path_id: discontinuous_path, ...},
            'average_time_step': description
        }
    """
    paths = gt_dict['paths']
    if path_idxs is None:
        path_idxs = list(paths.keys())
    
    discontinuous_paths = {}
    for path_i in path_idxs:
        discontinuous_paths[path_i] = prepare_discontinuous_path(
            paths[path_i], mask_ratio
        )
    
    return {
        'paths': discontinuous_paths,
        'average_time_step': f'mask_ratio={mask_ratio}'
    }


def prepare_sparse_observations(
    gt_dict: Dict,
    avi_ids: np.ndarray,
    path_idxs: Optional[List[int]] = None
) -> Dict:
    """
    Simulate sparse observation data from AVI (Automatic Vehicle Identification) detectors
    
    Args:
        gt_dict: Ground truth path dictionary
        avi_ids: Array of link IDs where AVI detectors are deployed
        path_idxs: List of path indices to process
    
    Returns:
        Sparse observation path dictionary
    """
    paths = gt_dict['paths']
    if path_idxs is None:
        path_idxs = list(paths.keys())
    
    sparse_paths = {}
    for path_i in path_idxs:
        path = paths[path_i]
        sparse_path = []
        
        for i in range(len(path)):
            # Keep origin, destination or links on AVI detectors
            if i == 0 or i == len(path) - 1 or path[i] in avi_ids:
                sparse_path.append(path[i])
        
        sparse_paths[path_i] = np.array(sparse_path)
    
    return {
        'paths': sparse_paths
    }


def train_test_split(
    gt_dict: Dict,
    train_ratio: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split path data into training and test sets
    
    Args:
        gt_dict: Ground truth path dictionary
        train_ratio: Training set ratio
    
    Returns:
        (train_idxs, test_idxs): Index arrays for training and test sets
    """
    path_num = len(list(gt_dict['paths'].keys()))
    per = np.random.permutation(path_num)
    train_num = int(path_num * train_ratio)
    train_idxs = per[:train_num]
    test_idxs = per[train_num:]
    return train_idxs, test_idxs

