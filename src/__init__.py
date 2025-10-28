"""
RoutesFormer is a sequence-to-sequence deep learning model designed to infer complete paths from sparse vehicle trajectory observations.
Built on the Transformer architecture, it can handle discontinuous paths and perform high-precision path completion.
"""

__version__ = '1.0.0'
__author__ = 'RoutesFormer Team'

from .data_loader import (
    prepare_discontinuous_path,
    prepare_training_samples,
    prepare_sparse_observations,
    train_test_split
)
from .network_preprocess import enrich_network_info
from .routesformer import RoutesFormer
from .models import RoutesFormerTransformer

__all__ = [
    'prepare_discontinuous_path',
    'prepare_training_samples',
    'prepare_sparse_observations',
    'train_test_split',
    'enrich_network_info',
    'RoutesFormer',
    'RoutesFormerTransformer',
]

