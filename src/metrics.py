"""
Path Inference Evaluation Metrics Module
Contains calculation of metrics such as BLEU, JSD, ED, TLLA
"""

import numpy as np
import scipy.stats
from typing import Dict, List, Tuple, Optional


def levenshtein_distance(path1: Tuple, path2: Tuple) -> int:
    """
    Calculate edit distance between two path sequences（Levenshtein Distance）
    
    Args:
        path1: First path sequence
        path2: Second path sequence
        
    Returns:
        Edit distance value
    """
    len1, len2 = len(path1), len(path2)
    matrix = [[i + j for j in range(len2 + 1)] for i in range(len1 + 1)]
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if path1[i-1] == path2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j] + 1,      # deletion
                             matrix[i][j-1] + 1,         # insertion
                             matrix[i-1][j-1] + d)       # substitution
    
    return matrix[len1][len2]


def calculate_bleu(reference_paths: List[Tuple], predicted_path: Tuple) -> float:
    """
    Calculate BLEU score(using 1-gram)
    
    Args:
        reference_paths: Reference path list
        predicted_path: Predicted path
        
    Returns:
        BLEUscore
    """
    from nltk.translate.bleu_score import sentence_bleu
    
    try:
        bleu = sentence_bleu(reference_paths, predicted_path, weights=(1, 0, 0, 0))
    except:
        bleu = 0.0
    
    return bleu


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate JS divergence（Jensen-Shannon Divergence）
    
    Args:
        p: Probability distribution 1
        q: Probability distribution 2
        
    Returns:
        JS divergence value
    """
    M = (p + q) / 2
    jsd = 0.5 * scipy.stats.entropy(p, M, base=2) + 0.5 * scipy.stats.entropy(q, M, base=2)
    return jsd


def calculate_path_bleu(
    gt_dict: Dict,
    s_dict: Dict,
    path_probabilities: Dict[Tuple, float],
    path_i: int,
    is_global_path: bool = True
) -> float:
    """
    Calculate BLEU score for single path
    
    Args:
        gt_dict: Ground truth path dictionary
        s_dict: sparse path dictionary
        path_probabilities: path probability dictionary {path: probability}
        path_i: Path index
        is_global_path: Whether it is a global path
        
    Returns:
        BLEUscore
    """
    if len(path_probabilities) == 0:
        return 0.0
    
    gt_path = tuple(gt_dict['paths'][path_i])
    if not is_global_path:
        s_gt_link_idx = s_dict['S_GT_link_idxs'][path_i]
        gt_path = gt_path[s_gt_link_idx[0]:s_gt_link_idx[-1] + 1]
    
    reference_paths = [gt_path]
    bleu_score = 0.0
    
    for pred_path, prob in path_probabilities.items():
        bleu = calculate_bleu(reference_paths, pred_path)
        bleu_score += prob * bleu
    
    return bleu_score


def calculate_paths_bleu(
    gt_dict: Dict,
    s_dict: Dict,
    all_path_probabilities: Dict[int, Dict[Tuple, float]],
    is_global_path: bool = True,
    idxs: Optional[List[int]] = None
) -> float:
    """
    Calculate average of multiple pathsBLEUscore
    
    Args:
        gt_dict: Ground truth path dictionary
        s_dict: sparse path dictionary
        all_path_probabilities: probability dictionary for all paths
        is_global_path: Whether it is a global path
        idxs: List of path indices to calculate
        
    Returns:
        Average BLEU score
    """
    bleu_scores = []
    
    for path_i, path_probs in all_path_probabilities.items():
        if idxs is None or path_i in idxs:
            bleu = calculate_path_bleu(gt_dict, s_dict, path_probs, path_i, is_global_path)
            if not np.isnan(bleu):
                bleu_scores.append(bleu)
            else:
                bleu_scores.append(0.0)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


def calculate_path_ed(
    gt_dict: Dict,
    s_dict: Dict,
    path_probabilities: Dict[Tuple, float],
    path_i: int,
    is_global_path: bool = True
) -> float:
    """
    Calculate normalized edit distance for single path
    
    Args:
        gt_dict: Ground truth path dictionary
        s_dict: sparse path dictionary
        path_probabilities: path probability dictionary
        path_i: Path index
        is_global_path: Whether it is a global path
        
    Returns:
        Normalized edit distance
    """
    if len(path_probabilities) == 0:
        return 1.0
    
    gt_path = tuple(gt_dict['paths'][path_i])
    if not is_global_path:
        s_gt_link_idx = s_dict['S_GT_link_idxs'][path_i]
        gt_path = gt_path[s_gt_link_idx[0]:s_gt_link_idx[-1] + 1]
    
    edit_distance = 0.0
    for pred_path, prob in path_probabilities.items():
        ed = levenshtein_distance(gt_path, pred_path)
        normalized_ed = min(ed / len(gt_path), 1.0)
        edit_distance += prob * normalized_ed
    
    return edit_distance


def calculate_paths_ed(
    gt_dict: Dict,
    s_dict: Dict,
    all_path_probabilities: Dict[int, Dict[Tuple, float]],
    is_global_path: bool = True,
    idxs: Optional[List[int]] = None
) -> float:
    """
    Calculate average edit distance of multiple paths
    
    Args:
        gt_dict: Ground truth path dictionary
        s_dict: sparse path dictionary
        all_path_probabilities: probability dictionary for all paths
        is_global_path: Whether it is a global path
        idxs: List of path indices to calculate
        
    Returns:
        Average edit distance
    """
    edit_distances = []
    
    for path_i, path_probs in all_path_probabilities.items():
        if idxs is None or path_i in idxs:
            ed = calculate_path_ed(gt_dict, s_dict, path_probs, path_i, is_global_path)
            if not np.isnan(ed):
                edit_distances.append(ed)
            else:
                edit_distances.append(1.0)
    
    return np.mean(edit_distances) if edit_distances else 1.0


def calculate_path_tlla(
    network,
    gt_dict: Dict,
    s_dict: Dict,
    path_probabilities: Dict[Tuple, float],
    path_i: int,
    is_global_path: bool = True
) -> float:
    """
    Calculate TLLA (Total Link Length Accuracy) for single path
    
    Args:
        network: road network graph
        gt_dict: Ground truth path dictionary
        s_dict: sparse path dictionary
        path_probabilities: path probability dictionary
        path_i: Path index
        is_global_path: Whether it is a global path
        
    Returns:
        TLLA value
    """
    if len(path_probabilities) == 0:
        return 0.0
    
    gt_path = tuple(gt_dict['paths'][path_i])
    if not is_global_path:
        s_gt_link_idx = s_dict['S_GT_link_idxs'][path_i]
        gt_path = gt_path[s_gt_link_idx[0]:s_gt_link_idx[-1] + 1]
    
    link_nodes_dict = network.graph['link_nodes_dict']
    
    # Calculate total length of ground truth path
    gt_length = 0.0
    for link in gt_path:
        node1, node2 = link_nodes_dict[link]
        gt_length += network.edges[node1, node2]['LENGTH']
    
    # Calculate TLLA
    tlla = 0.0
    for pred_path, prob in path_probabilities.items():
        matched_length = 0.0
        unique_pred_path = tuple(set(pred_path))
        
        for link in unique_pred_path:
            if link in gt_path:
                node1, node2 = link_nodes_dict[link]
                matched_length += network.edges[node1, node2]['LENGTH']
        
        tlla += prob * (matched_length / gt_length)
    
    return tlla


def calculate_paths_tlla(
    network,
    gt_dict: Dict,
    s_dict: Dict,
    all_path_probabilities: Dict[int, Dict[Tuple, float]],
    is_global_path: bool = True,
    idxs: Optional[List[int]] = None
) -> float:
    """
    Calculate average of multiple pathsTLLA
    
    Args:
        network: road network graph
        gt_dict: Ground truth path dictionary
        s_dict: sparse path dictionary
        all_path_probabilities: probability dictionary for all paths
        is_global_path: Whether it is a global path
        idxs: List of path indices to calculate
        
    Returns:
        Average TLLA value
    """
    tlla_values = []
    
    for path_i, path_probs in all_path_probabilities.items():
        if idxs is None or path_i in idxs:
            tlla = calculate_path_tlla(network, gt_dict, s_dict, path_probs, path_i, is_global_path)
            if not np.isnan(tlla):
                tlla_values.append(tlla)
            else:
                tlla_values.append(0.0)
    
    return np.mean(tlla_values) if tlla_values else 0.0


def calculate_paths_jsd(
    gt_dict: Dict,
    s_dict: Dict,
    all_path_probabilities: Dict[int, Dict[Tuple, float]],
    is_global_path: bool = True,
    idxs: Optional[List[int]] = None,
    include_unseen: bool = True
) -> float:
    """
    Calculate path distribution JS divergence
    
    Args:
        gt_dict: Ground truth path dictionary
        s_dict: sparse path dictionary
        all_path_probabilities: probability dictionary for all paths
        is_global_path: Whether it is a global path
        idxs: List of path indices to calculate
        include_unseen: include paths that failed to connect or not
        
    Returns:
        JS divergence value
    """
    gt_route_freq = {}
    pred_route_freq = {}
    
    for path_i, path_probs in all_path_probabilities.items():
        if idxs is None or path_i in idxs:
            gt_path = tuple(gt_dict['paths'][path_i])
            if not is_global_path:
                s_gt_link_idx = s_dict['S_GT_link_idxs'][path_i]
                gt_path = gt_path[s_gt_link_idx[0]:s_gt_link_idx[-1] + 1]
            
            if gt_path not in gt_route_freq:
                gt_route_freq[gt_path] = 0
            gt_route_freq[gt_path] += 1
            
            if len(path_probs) == 0 and include_unseen:
                # Paths that failed to connect
                unseen_key = 'unseen'
                if unseen_key not in pred_route_freq:
                    pred_route_freq[unseen_key] = 0
                pred_route_freq[unseen_key] += 1
            
            for pred_path, prob in path_probs.items():
                if pred_path not in pred_route_freq:
                    pred_route_freq[pred_path] = 0
                pred_route_freq[pred_path] += prob
    
    # Merge all paths into one dictionary中
    total_route_freq = {}
    for route in gt_route_freq:
        if route not in total_route_freq:
            total_route_freq[route] = [0, 0]
        total_route_freq[route][0] = gt_route_freq[route]
    
    for route in pred_route_freq:
        if route not in total_route_freq:
            total_route_freq[route] = [0, 0]
        total_route_freq[route][1] = pred_route_freq[route]
    
    # Build distribution array
    gt_distribution = []
    pred_distribution = []
    for route in total_route_freq:
        gt_distribution.append(total_route_freq[route][0])
        pred_distribution.append(total_route_freq[route][1])
    
    gt_distribution = np.array(gt_distribution, dtype=float)
    pred_distribution = np.array(pred_distribution, dtype=float)
    
    # normalize
    gt_distribution = gt_distribution / np.sum(gt_distribution)
    pred_distribution = pred_distribution / np.sum(pred_distribution)
    
    # Calculate JS divergence
    jsd = js_divergence(gt_distribution, pred_distribution)
    
    return jsd


def evaluate_all_metrics(
    network,
    gt_dict: Dict,
    s_dict: Dict,
    all_path_probabilities: Dict[int, Dict[Tuple, float]],
    is_global_path: bool = True,
    idxs: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Args:
        network: road network graph
        gt_dict: Ground truth path dictionary
        s_dict: sparse path dictionary
        all_path_probabilities: probability dictionary for all paths
        is_global_path: Whether it is a global path
        idxs: List of path indices to calculate
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # BLEU
    metrics['BLEU'] = calculate_paths_bleu(
        gt_dict, s_dict, all_path_probabilities, is_global_path, idxs
    )
    
    # ED (edit distance)
    metrics['ED'] = calculate_paths_ed(
        gt_dict, s_dict, all_path_probabilities, is_global_path, idxs
    )
    
    # TLLA
    metrics['TLLA'] = calculate_paths_tlla(
        network, gt_dict, s_dict, all_path_probabilities, is_global_path, idxs
    )
    
    # JSD (include paths that failed to connect)
    metrics['JSD'] = calculate_paths_jsd(
        gt_dict, s_dict, all_path_probabilities, is_global_path, idxs, include_unseen=True
    )
    
    # JSD (exclude paths that failed to connect)
    metrics['JSD_without_unseen'] = calculate_paths_jsd(
        gt_dict, s_dict, all_path_probabilities, is_global_path, idxs, include_unseen=False
    )
    
    return metrics

