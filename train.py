"""
RoutesFormer Training Script

Train RoutesFormer model from scratch
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
    ModelConfig, ExperimentConfig, MethodConfig, TrainConfig,
    NETWORK_FILE, GT_DICT_FILE, MODEL_DIR
)
from src.network_preprocess import enrich_network_info
from src.data_loader import train_test_split
from src.routesformer import RoutesFormer
from src.utils import setup_logger, ensure_dir

warnings.filterwarnings("ignore")

# Setup logger
logger = setup_logger('RoutesFormer-Train', logging.INFO)


def main():
    """Main training process"""
    logger.info("="*60)
    logger.info("RoutesFormer Training Script")
    logger.info("="*60)
    
    # Create necessary directories
    ensure_dir(MODEL_DIR)
    
    # ========== 1. Load Road Network ==========
    logger.info("\nStep 1/5: Loading road network data...")
    if not os.path.exists(NETWORK_FILE):
        logger.error(f"Network file not found: {NETWORK_FILE}")
        logger.error("Please ensure data files are in the data directory")
        return
    
    network = nx.read_gml(NETWORK_FILE)
    logger.info(f"Network loaded successfully, contains {len(network.edges)} links")
    
    # ========== 2. Preprocess Network ==========
    logger.info("\nStep 2/5: Preprocessing network information...")
    network = enrich_network_info(network, data_dir='data')
    logger.info(f"Network preprocessing completed")
    
    # ========== 3. Load Training Data ==========
    logger.info("\nStep 3/5: Loading ground truth path data...")
    if not os.path.exists(GT_DICT_FILE):
        logger.error(f"Ground truth path file not found: {GT_DICT_FILE}")
        return
    
    GT_dict = np.load(GT_DICT_FILE, allow_pickle=True).item()
    logger.info(f"Number of ground truth paths: {len(GT_dict['paths'])}")
    
    # Statistics of path lengths
    max_len = -1
    min_len = 100
    for path_i in GT_dict['paths']:
        path = GT_dict['paths'][path_i]
        if len(path) > max_len:
            max_len = len(path)
        if len(path) < min_len:
            min_len = len(path)
    logger.info(f"Path length range: {min_len} ~ {max_len}")
    
    # Statistics of maximum number of adjacent links
    max_alter_link_num = -1
    for link_O in network.graph['neighbor_links_O']:
        if len(network.graph['neighbor_links_O'][link_O]) > max_alter_link_num:
            max_alter_link_num = len(network.graph['neighbor_links_O'][link_O])
    logger.info(f"Maximum number of adjacent links: {max_alter_link_num}")
    
    # ========== 4. Split Train/Test Sets ==========
    logger.info("\nStep 4/5: Splitting train and test sets...")
    
    if TrainConfig.is_load_idx:
        train_idxs = np.load(os.path.join(MODEL_DIR, 'train_idxs.npy'))
        test_idxs = np.load(os.path.join(MODEL_DIR, 'test_idxs.npy'))
        logger.info("Loaded train/test indices from file")
    else:
        train_idxs, test_idxs = train_test_split(
            GT_dict,
            train_ratio=TrainConfig.train_test_split_rate
        )
        logger.info(f"Generated new train/test indices")
    
    logger.info(f"Training set size: {len(train_idxs)}")
    logger.info(f"Test set size: {len(test_idxs)}")
    
    if TrainConfig.is_save_idx:
        np.save(os.path.join(MODEL_DIR, 'train_idxs.npy'), train_idxs)
        np.save(os.path.join(MODEL_DIR, 'test_idxs.npy'), test_idxs)
        logger.info("Train/test indices saved")
    
    # ========== 5. Train Model ==========
    logger.info("\nStep 5/5: Training RoutesFormer model...")
    logger.info("-" * 60)
    logger.info("Model Configuration:")
    logger.info(f"  - Embedding size: {ModelConfig.embedding_size}")
    logger.info(f"  - Encoder layers: {ModelConfig.num_encoder_layers}")
    logger.info(f"  - Decoder layers: {ModelConfig.num_decoder_layers}")
    logger.info(f"  - Batch size: {ModelConfig.batch_size}")
    logger.info(f"  - Learning rate: {ModelConfig.learning_rate}")
    logger.info(f"  - Training epochs: {ModelConfig.epoch_num}")
    logger.info("-" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  - Max sequence length: {ExperimentConfig.max_len}")
    logger.info(f"  - Mask ratios: {ExperimentConfig.train_mask_ratios}")
    if TrainConfig.use_iterative_training:
        logger.info(f"  - Iterative training: Enabled")
        logger.info(f"  - Data regeneration interval: {TrainConfig.data_regeneration_interval} epochs")
        logger.info(f"  - Number of iterations: {TrainConfig.num_iterations}")
        logger.info(f"  - Total training epochs: {TrainConfig.data_regeneration_interval * TrainConfig.num_iterations}")
    else:
        logger.info(f"  - Iterative training: Disabled")
    logger.info("-" * 60)
    
    # Check if loading existing model
    model_path = os.path.join(MODEL_DIR, TrainConfig.model_name)
    if TrainConfig.is_load_model and os.path.exists(model_path):
        logger.info(f"Loading existing model: {model_path}")
        routes_former = torch.load(model_path, weights_only=False)
    else:
        logger.info("Initializing new model...")
        routes_former = RoutesFormer(
            network,
            ExperimentConfig,
            ModelConfig,
            MethodConfig
        )
        
        # Start training
        start_time = datetime.now()
        logger.info(f"Training start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        routes_former.train(
            network, 
            GT_dict, 
            train_idxs,
            use_iterative=TrainConfig.use_iterative_training,
            data_regen_interval=TrainConfig.data_regeneration_interval,
            num_iterations=TrainConfig.num_iterations
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Training duration: {duration}")
    
    # ========== 6. Save Model ==========
    if TrainConfig.is_save_model:
        logger.info(f"\nSaving model to: {model_path}")
        torch.save(routes_former, model_path)
        logger.info("Model saved successfully!")
    
    logger.info("\n" + "="*60)
    logger.info("Training process completed!")
    logger.info("="*60)
    logger.info(f"\nTips:")
    logger.info(f"  - Trained model saved at: {model_path}")
    logger.info(f"  - Run test.py to perform path inference testing")
    logger.info(f"  - Train/test indices saved at: {MODEL_DIR}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nError occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()

