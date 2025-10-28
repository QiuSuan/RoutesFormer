"""
RoutesFormer Testing Script

Perform path inference testing using trained model
"""
import os
import sys
import torch
import numpy as np
import networkx as nx
import warnings
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    TestConfig, NETWORK_FILE, GT_DICT_FILE, MODEL_DIR
)
from src.network_preprocess import enrich_network_info
from src.data_loader import prepare_sparse_observations
from src.utils import setup_logger, is_path_subsequence_of_path
from src.metrics import evaluate_all_metrics

warnings.filterwarnings("ignore")

# Setup logger
logger = setup_logger('RoutesFormer-Test', logging.INFO)


def calculate_metrics(network, predicted_paths, GT_dict, S_dict, test_idxs):
    """
    Calculate evaluation metrics (BLEU, JSD, ED, TLLA)
    
    Args:
        network: Road network graph
        predicted_paths: Predicted paths dictionary
        GT_dict: Ground truth paths dictionary
        S_dict: Sparse observations dictionary
        test_idxs: Test indices
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate path completion rate
    total_paths = len(test_idxs)
    completed_paths = sum(1 for path_i in test_idxs if path_i in predicted_paths and len(predicted_paths[path_i]) > 0)
    path_completion_rate = completed_paths / total_paths if total_paths > 0 else 0
    
    # Use new evaluation metrics module
    metrics = evaluate_all_metrics(
        network=network,
        gt_dict=GT_dict,
        s_dict=S_dict,
        all_path_probabilities=predicted_paths,
        is_global_path=True,
        idxs=test_idxs
    )
    
    # Add path completion rate statistics
    metrics['path_completion_rate'] = path_completion_rate
    metrics['total_paths'] = total_paths
    metrics['completed_paths'] = completed_paths
    
    return metrics


def main():
    """Main testing process"""
    logger.info("="*60)
    logger.info("RoutesFormer Testing Script")
    logger.info("="*60)
    
    # ========== 1. Load Road Network ==========
    logger.info("\nStep 1/6: Loading road network data...")
    if not os.path.exists(NETWORK_FILE):
        logger.error(f"Network file not found: {NETWORK_FILE}")
        return
    
    network = nx.read_gml(NETWORK_FILE)
    logger.info(f"Network loaded successfully, contains {len(network.edges)} links")
    
    # ========== 2. Preprocess Network ==========
    logger.info("\nStep 2/6: Preprocessing network information...")
    network = enrich_network_info(network, data_dir='data')
    logger.info("Network preprocessing completed")
    
    # ========== 3. Load Ground Truth Paths ==========
    logger.info("\nStep 3/6: Loading ground truth path data...")
    if not os.path.exists(GT_DICT_FILE):
        logger.error(f"Ground truth path file not found: {GT_DICT_FILE}")
        return
    
    GT_dict = np.load(GT_DICT_FILE, allow_pickle=True).item()
    logger.info(f"Number of ground truth paths: {len(GT_dict['paths'])}")
    
    # ========== 4. Load Dataset Indices ==========
    logger.info("\nStep 4/6: Loading dataset indices...")
    
    # Select test or train dataset based on configuration
    if TestConfig.test_on_dataset == 'train':
        idx_file = os.path.join(MODEL_DIR, 'train_idxs.npy')
        dataset_name = 'Training Set'
        logger.info(f"Note: Testing on training set (for overfitting check)")
    else:
        idx_file = os.path.join(MODEL_DIR, 'test_idxs.npy')
        dataset_name = 'Test Set'
    
    if not os.path.exists(idx_file):
        logger.error(f"{dataset_name} index file not found: {idx_file}")
        logger.error("Please run train.py first to generate train/test split")
        return
    
    test_idxs = np.load(idx_file)
    
    logger.info(f"Using dataset: {dataset_name}")
    logger.info(f"Number of test paths: {len(test_idxs)} (using all)")
    
    # ========== 5. Simulate AVI Detector Deployment ==========
    logger.info("\nStep 5/6: Simulating AVI detector deployment...")
    logger.info(f"AVI coverage: {TestConfig.avi_coverage * 100:.1f}%")
    
    # Randomly deploy AVI detectors
    AVI_num = int(TestConfig.avi_coverage * len(network.edges))
    AVI_ids = np.random.permutation(len(network.edges))[:AVI_num]
    logger.info(f"Deployed {AVI_num} AVI detectors")
    
    # Generate sparse observations
    S_dict = prepare_sparse_observations(GT_dict, AVI_ids, test_idxs)
    
    # Calculate sparsity statistics
    total_obs = 0
    total_gt = 0
    for path_i in test_idxs:
        total_obs += len(S_dict['paths'][path_i])
        total_gt += len(GT_dict['paths'][path_i])
    
    obs_rate = total_obs / total_gt if total_gt > 0 else 0
    logger.info(f"Actual observation rate: {obs_rate * 100:.2f}%")
    
    # ========== 6. Load Model and Perform Inference ==========
    logger.info("\nStep 6/6: Loading model and performing path inference...")
    
    model_path = TestConfig.model_file
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run train.py first to train the model")
        return
    
    logger.info(f"Loading model: {model_path}")
    routes_former = torch.load(model_path, weights_only=False)
    routes_former.use_shortest_path = TestConfig.use_shortest_path
    
    logger.info("Starting path inference...")
    start_time = datetime.now()
    
    predicted_paths = {}
    success_count = 0
    fail_count = 0
    
    for idx, path_i in enumerate(test_idxs):
        try:
            path_prob = routes_former.predict_path(network, S_dict, path_i)
            predicted_paths[path_i] = path_prob
            
            if len(path_prob) > 0:
                success_count += 1
            else:
                fail_count += 1
                if idx < 5:  # Only log first few failure cases
                    logger.debug(f"Path {path_i} prediction failed (returned empty dict)")
                    logger.debug(f"  Observation path: {S_dict['paths'][path_i]}")
        except Exception as e:
            fail_count += 1
            logger.warning(f"Path {path_i} inference exception: {str(e)}")
            predicted_paths[path_i] = {}
            if idx < 5:
                import traceback
                logger.debug(traceback.format_exc())
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(test_idxs)}, Success: {success_count}, Fail: {fail_count}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Inference completed, duration: {duration}")
    logger.info(f"Successful predictions: {success_count}/{len(test_idxs)} ({success_count/len(test_idxs)*100:.2f}%)")
    logger.info(f"Failed predictions: {fail_count}/{len(test_idxs)} ({fail_count/len(test_idxs)*100:.2f}%)")
    
    # ========== 7. Evaluate Results ==========
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)
    
    # Only calculate metrics when there are successful predictions
    if success_count == 0:
        logger.warning("All path predictions failed, cannot calculate evaluation metrics")
        logger.warning("Possible reasons:")
        logger.warning("  1. Model not properly trained")
        logger.warning("  2. AVI coverage too low")
        logger.warning("  3. Model configuration does not match test data")
        return
    
    metrics = calculate_metrics(network, predicted_paths, GT_dict, S_dict, test_idxs)
    
    logger.info(f"\nPath Completion Rate:")
    logger.info(f"  - nPath Completion rate: {metrics['path_completion_rate']:.4f} ({metrics['completed_paths']}/{metrics['total_paths']})")
    
    logger.info(f"\nPath Inference Quality Metrics:")
    logger.info(f"  - BLEU score: {metrics['BLEU']:.4f} (higher is better)")
    logger.info(f"  - Edit Distance (ED): {metrics['ED']:.4f} (lower is better)")
    logger.info(f"  - Total Link Length Accuracy (TLLA): {metrics['TLLA']:.4f} (higher is better)")
    logger.info(f"  - JS Divergence (JSD): {metrics['JSD']:.4f} (lower is better)")
    logger.info(f"  - JS Divergence without unseen: {metrics['JSD_without_unseen']:.4f} (lower is better)")
    
    # Save results
    dataset_suffix = 'train' if TestConfig.test_on_dataset == 'train' else 'test'
    result_file = os.path.join(MODEL_DIR, f'{dataset_suffix}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy')
    np.save(result_file, {
        'predicted_paths': predicted_paths,
        'metrics': metrics,
        'test_idxs': test_idxs,
        'avi_coverage': TestConfig.avi_coverage,
        'dataset': TestConfig.test_on_dataset
    })
    logger.info(f"\nTest results saved to: {result_file}")
    
    logger.info("\n" + "="*60)
    logger.info("Testing completed!")
    logger.info("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTesting interrupted by user")
    except Exception as e:
        logger.error(f"\nError occurred during testing: {str(e)}")
        import traceback
        traceback.print_exc()

