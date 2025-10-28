# RoutesFormer

**A Transformer-based Path Inference and Route Choice Model**

RoutesFormer is a sequence-to-sequence deep learning model designed to infer complete paths from sparse vehicle trajectory observations. Built on the Transformer architecture, it can handle discontinuous paths and perform high-precision path completion.

## Project Overview

This project implements the method proposed in the paper "RoutesFormer: A sequence-based route choice Transformer for efficient path inference from sparse trajectories". Key features include:

- **Path Inference**: Infer complete paths from sparse link observations
- **Route Choice Modeling**: Learn and predict route choice behavior
- **End-to-End Training**: Unified sequence-to-sequence framework
- **Attention Mechanism**: Interpretable route choice attention

## Installation

### 1. Clone or Download the Project

```bash
cd RoutesFormer
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n routesformer python=3.9
conda activate routesformer
```

### 3. Install Dependencies

**(Optional) Change pip source for faster installation if encountering network issues:**

```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

Install required packages:

```bash
pip install -r requirements.txt
```

If GPU support is needed, install PyTorch according to your CUDA version:

```bash
# Example: CUDA 11.8
pip install torch==2.3.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Preparation

### Option 1: Using Simulated Data (Recommended for Quick Start)

**No real data needed** - simply run the data generation script:

```bash
python generate_sample_data.py
```

This script will automatically generate:
- **road_network.gml**: 10×10 grid network (adjustable)
- **GT_dict.npy**: 300 random paths (adjustable)
- **neighbor_links_O.npy / neighbor_links_D.npy**: Link adjacency relationships
- **network_visualization.png**: Network visualization

**Configuration Parameters** (modifiable in the script):
```python
M = 10                  # Grid rows
N = 10                  # Grid columns
NUM_PATHS = 300         # Number of paths
MIN_PATH_LENGTH = 8     # Minimum path length
MAX_PATH_LENGTH = 25    # Maximum path length
```

After generation, simply run `python train.py` to start training!

### Option 2: Using Real Data

If you have real road network and trajectory data, the following data files should be placed in the `data/` directory:

```
data/
├── road_network.gml              # Road network file (GML format)
├── GT_dict.npy                   # Ground truth path dictionary
├── neighbor_links_O.npy          # Downstream link adjacency
├── neighbor_links_D.npy          # Upstream link adjacency
```

### Data Format Specification

#### 1. Road Network File (road_network.gml)

NetworkX-formatted directed graph with the following node and edge attributes:

- **Node Attributes**:
  - `LON`: Longitude
  - `LAT`: Latitude

- **Edge Attributes**:
  - `EDGEID`: Link ID (integer)
  - `LENGTH`: Link length
  - `VLENGTH`: Virtual length (for shortest path computation)

#### 2. Ground Truth Path Dictionary (GT_dict.npy)

Dictionary containing complete trajectories:

```python
{
    'paths': {
        0: [link1, link2, link3, ...],  # Link sequence for path 0
        1: [link4, link5, link6, ...],  # Link sequence for path 1
        ...
    }
}
```

#### 3. Adjacency Relationship Files

- **neighbor_links_O.npy**: Dictionary where key is link ID and value is list of downstream adjacent link IDs
- **neighbor_links_D.npy**: Dictionary where key is link ID and value is list of upstream adjacent link IDs

## Usage

### Quick Start (3 Steps)

#### Step 0: Generate Simulated Data (First Run)

```bash
python generate_sample_data.py
```

This will generate all necessary data files in the `data/` directory. Check the generated `data/network_visualization.png` to confirm the network was created correctly.

#### Step 1: Train the Model

```bash
python train.py
```

The training process will:
- Load road network and ground truth path data
- Automatically split training and test sets (default 5% for training)
- Train the RoutesFormer model
- Save the trained model to `models/RoutesFormer_cloze.pth`


#### Step 2: Test the Model

```bash
python test.py
```

The testing process will:
- Load the trained model
- Simulate AVI detector deployment (default 20% coverage)
- Perform path inference on the test set
- Calculate evaluation metrics and save results


### Configuration Parameters

All configuration parameters are defined in `config.py` and can be modified as needed:

#### Model Configuration (ModelConfig)

```python
embedding_size = 64          # Embedding dimension
num_encoder_layers = 2       # Number of encoder layers
num_decoder_layers = 2       # Number of decoder layers
nhead = 8                    # Number of attention heads
batch_size = 256             # Batch size
learning_rate = 1e-4         # Learning rate
epoch_num = 500              # Number of training epochs
```

#### Training Configuration (TrainConfig)

```python
train_test_split_rate = 0.80           # Train/test split ratio (1.0 = all training)
# Iterative Training (Data Augmentation)
use_iterative_training = True          # Enable iterative training
data_regeneration_interval = 100       # Regenerate data every N epochs
num_iterations = 10                    # Total iterations (epochs = interval × iterations)
```

**Iterative Training**: Periodically regenerates training data with different random masks to improve generalization.

#### Experiment Configuration (ExperimentConfig)

```python
max_len = 30                                     # Maximum sequence length
train_mask_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 'OD']  # Masking ratios
```

#### Test Configuration (TestConfig)

```python
avi_coverage = 0.4              # AVI detector coverage (tests all paths)
use_shortest_path = False       # Use shortest path as fallback when model prediction fails
test_on_dataset = 'train'       # 'train' or 'test' dataset
```

**Note on `use_shortest_path`**: When `True`, if model prediction fails (cannot produce a valid path covering all observed links), the shortest path between observed links will be used as fallback. If model prediction succeeds, the model output will be used instead.

## Project Structure

```
RoutesFormer/
├── data/                           # Data directory
│   ├── road_network.gml           # Road network file
│   ├── GT_dict.npy                # Ground truth paths
│   ├── neighbor_links_O.npy       # Downstream adjacency
│   ├── neighbor_links_D.npy       # Upstream adjacency
│   ├── network_visualization.png  # Network visualization (generated)
│   └── preprocessed/              # Preprocessing results
│
├── src/                           # Source code directory
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # Data loading module
│   ├── network_preprocess.py     # Network preprocessing
│   ├── models.py                 # Transformer models
│   ├── routesformer.py           # RoutesFormer main class
│   └── utils.py                  # Utility functions
│
├── models/                        # Model save directory
│   ├── RoutesFormer_cloze.pth    # Trained model
│   ├── train_idxs.npy            # Training set indices
│   └── test_idxs.npy             # Test set indices
│
├── generate_sample_data.py       # Data generation script
├── train.py                       # Training script
├── test.py                        # Testing script
├── config.py                      # Configuration file
├── requirements.txt               # Dependencies list
└── README.md                      # This file
```

## Core Modules Description

### 1. data_loader.py

Data loading and preprocessing module providing the following functions:

- `prepare_discontinuous_path()`: Generate discontinuous paths from complete paths
- `prepare_training_samples()`: Prepare training samples
- `prepare_sparse_observations()`: Simulate sparse observations
- `train_test_split()`: Dataset splitting

### 2. network_preprocess.py

Road network preprocessing module:

- `enrich_network_info()`: Enrich network information and add adjacency relationships
- `construct_twin_network()`: Construct twin network
- `get_candidate_paths_k_shortest()`: K-shortest path search

### 3. models.py

Transformer model implementation:

- `PositionalEncoding`: Positional encoding layer
- `TransformerModel`: Core Transformer model
- `RoutesFormerTransformer`: Model wrapper class

### 4. routesformer.py

RoutesFormer main methods:

- `DataGenerator`: Data generator
- `RoutesFormer`: Main class providing training and inference interfaces

## Citation

If you use this project, please cite the original paper:

```bibtex
@article{qiu2024routesformer,
  title={RoutesFormer: A sequence-based route choice Transformer for efficient path inference from sparse trajectories},
  author={Qiu, Shuhan and Qin, Guoyang and Wong, Melvin and Sun, Jian},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={162},
  pages={104552},
  year={2024},
  publisher={Elsevier}
}
```

## License

This project is for academic research use only.

## Changelog

### v1.0.0 (2025-10-28)

- Initial release
- Implemented complete training and testing pipeline
- Code refactoring for improved readability and maintainability
- Added detailed documentation
- Added simulated data generation feature for quick start without real data

## Acknowledgments

Thanks to the authors of the original paper and all contributors.

