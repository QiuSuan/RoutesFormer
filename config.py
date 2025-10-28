"""
RoutesFormer Configuration File

Contains all configuration parameters for training and testing
"""
import os

# ==================== Path Configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data file paths
NETWORK_FILE = os.path.join(DATA_DIR, 'road_network.gml')
GT_DICT_FILE = os.path.join(DATA_DIR, 'GT_dict.npy')
NEIGHBOR_LINKS_O_FILE = os.path.join(DATA_DIR, 'neighbor_links_O.npy')
NEIGHBOR_LINKS_D_FILE = os.path.join(DATA_DIR, 'neighbor_links_D.npy')
NETWORK_SHORTEST_PATHS_FILE = os.path.join(DATA_DIR, 'network_shortest_paths.npy')

# ==================== Model Configuration ====================
class ModelConfig:
    """Model hyperparameter configuration"""
    # Model architecture
    embedding_size = 64
    num_encoder_layers = 3
    num_decoder_layers = 3
    nhead = 8
    dim_feedforward = 512
    
    # Training parameters
    batch_size = 256
    learning_rate = 1e-4
    epoch_num = 500
    
    # Positional encoding
    train_positional_encoding = True
    eval_positional_encoding = True
    
    # Attention mask
    train_decoder_masked = False
    eval_decoder_masked = False

# ==================== Experiment Configuration ====================
class ExperimentConfig:
    """Experiment settings configuration"""
    # Sequence length
    max_len = 30
    
    # Mask settings
    is_mask_token = True
    
    # Training mask ratios
    train_mask_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 'OD']
    
    # Attribute dictionary
    attributes_dict = {'onehot_embedding': -1}

# ==================== Method Configuration ====================
class MethodConfig:
    """Inference method configuration"""
    # Path generation method
    path_generation_method = 'argmax'  # 'argmax', 'shortest_path'
    
    # Graph constraint
    graph_constraint = True
    
    # Discontinuous path attention type
    discontinuous_path_attention = 'global'
    
    # Whether to allow decoupled trying
    allow_decoupled_trying = False

# ==================== Training Configuration ====================
class TrainConfig:
    """Training process configuration"""
    # Dataset split
    train_test_split_rate = 0.80  # Training set ratio (80% for train, 20% for test)
    
    # Model saving
    model_save_dir = MODEL_DIR
    model_name = 'RoutesFormer_cloze.pth'
    
    # Whether to load/save
    is_load_model = False
    is_save_model = True
    is_load_idx = False
    is_save_idx = True
    
    # Log output interval
    log_interval = 20
    
    # Iterative training configuration (data augmentation)
    use_iterative_training = True  # Whether to enable iterative training
    data_regeneration_interval = 100  # Regenerate training data every N epochs
    num_iterations = 10  # Total number of iterations (data regeneration times)

# ==================== Test Configuration ====================
class TestConfig:
    """Test process configuration"""
    # AVI detector coverage
    avi_coverage = 0.4
    
    # Model loading
    model_file = os.path.join(MODEL_DIR, 'RoutesFormer_cloze.pth')
    
    # Whether to use shortest path as fallback
    use_shortest_path = False
    
    # Test dataset selection: 'test' - use test set, 'train' - use train set (for overfitting check)
    test_on_dataset = 'train'  # Options: 'test' or 'train'

# ==================== Create Necessary Directories ====================
def create_directories():
    """Create necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'preprocessed'), exist_ok=True)

if __name__ == '__main__':
    create_directories()
    print("Configuration loaded")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")

