"""
RoutesFormer Main Methods Module

Contains data generator and main path inference methods for RoutesFormer model
"""
import numpy as np
import torch
import copy
import logging
from typing import Dict, List, Tuple, Optional
from .data_loader import prepare_training_samples
from .models import RoutesFormerTransformer
from .utils import is_path_subsequence_of_path, get_num_links

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Data generator base class
    
    Responsible for converting path data into input format acceptable to the model
    """
    def __init__(self, network, experiment_config, model_config, method_config):
        self.max_len = experiment_config.max_len
        self.train_mask_ratios = experiment_config.train_mask_ratios
        self.is_mask_token = experiment_config.is_mask_token
        self.attributes_dict = experiment_config.attributes_dict
        
        # Special token definitions
        self.tokens = ['bos', 'eos', 'mos', 'pos']  # begin, end, mask, padding
        self.token_indexs = {}
        for i, token in enumerate(self.tokens):
            self.token_indexs[token] = get_num_links(network) + i
        
        self.is_onehot_embedding = 'onehot_embedding' not in self.attributes_dict
    
    def construct_point_info(self, link, network, origin_sub_path=None, GPS_point=None):
        """
        Construct feature information for a link
        
        Args:
            link: Link ID or special token
            network: Road network object
            origin_sub_path: Original sub-path (optional)
            GPS_point: GPS point (optional)
        
        Returns:
            Link feature vector
        """
        point_info = []
        
        if link in self.token_indexs:
            # Special token
            if self.is_onehot_embedding:
                onehot_embedding = np.zeros(get_num_links(network) + len(self.tokens))
                onehot_embedding[self.token_indexs[link]] = 1
                point_info.extend(list(onehot_embedding))
            else:
                point_info.append(self.token_indexs[link])
        else:
            # Regular link
            if self.is_onehot_embedding:
                onehot_embedding = np.zeros(get_num_links(network) + len(self.tokens))
                onehot_embedding[link] = 1
                point_info.extend(list(onehot_embedding))
            else:
                point_info.append(link)
        
        point_info.append(0)  # Reserved attribute position
        return point_info
    
    def encode_discontinuous_path(self, network, S_dict, path_i):
        """
        Encode discontinuous path as model input sequence
        
        Args:
            network: Road network object
            S_dict: Sparse observations dictionary
            path_i: Path index
        
        Returns:
            Encoded path sequence
        """
        neighbor_links_O = network.graph['neighbor_links_O']
        S_path = tuple(list(S_dict['paths'][path_i]))
        path_src = []
        
        # Add begin token
        point_info = self.construct_point_info('bos', network, None, None)
        path_src.append(point_info)
        
        # Add observed links
        for i in range(len(S_path) - 1):
            point_info = self.construct_point_info(S_path[i], network, None, None)
            path_src.append(point_info)
            
            # If next link is not adjacent, add mask token
            try:
                if S_path[i + 1] not in neighbor_links_O.get(S_path[i], []):
                    point_info = self.construct_point_info('mos', network, None, None)
                    path_src.append(point_info)
            except:
                logger.warning(f"Link connection check failed: {S_path[i]} -> {S_path[i+1]}")
        
        # Add last link
        point_info = self.construct_point_info(S_path[-1], network, None, None)
        path_src.append(point_info)
        
        # Add end token
        point_info = self.construct_point_info('eos', network, None, None)
        path_src.append(point_info)
        
        # Pad to maximum length
        while len(path_src) < self.max_len:
            point_info = self.construct_point_info('pos', network, None, None)
            path_src.append(point_info)
        
        return np.array(path_src)
    
    def create_sequence_target(self, GT_dict, path_i):
        """
        Create target sequence (for training)
        
        Args:
            GT_dict: Ground truth path dictionary
            path_i: Path index
        
        Returns:
            Target sequence
        """
        GT_path = GT_dict['paths'][path_i]
        path_tgt_y = []
        
        # Begin token
        path_tgt_y.append(self.token_indexs['bos'])
        
        # Path sequence
        path_tgt_y.extend(GT_path)
        
        # End token
        path_tgt_y.append(self.token_indexs['eos'])
        
        # Padding
        while len(path_tgt_y) < self.max_len:
            path_tgt_y.append(self.token_indexs['pos'])
        
        return np.array(path_tgt_y)
    
    def create_dataset_once(self, network, GT_dict, path_idxs):
        """
        Create all training data at once
        
        Args:
            network: Road network object
            GT_dict: Ground truth path dictionary
            path_idxs: List of path indices
        
        Returns:
            (train_srcs, train_tgts, train_tgts_y)
        """
        from .data_loader import prepare_training_samples
        
        S_dicts = {}
        
        for mask_ratio_i, mask_ratio in enumerate(self.train_mask_ratios):
            logger.info(f"Generating training samples: {mask_ratio}_{mask_ratio_i}")
            S_dict = prepare_training_samples(GT_dict, mask_ratio, path_idxs)
            S_dicts[f"{mask_ratio}_{mask_ratio_i}"] = S_dict
        
        train_srcs, train_tgts, train_tgts_y = (), (), ()
        for key in S_dicts:
            print(key)
            S_dict = S_dicts[key]
            path_srcs, path_tgts, path_tgts_y = self._create_samples_from_dict(
                network, GT_dict, S_dict, path_idxs
            )
            train_srcs += (path_srcs,)
            train_tgts += (path_tgts,)
            train_tgts_y += (path_tgts_y,)
        
        train_srcs = np.concatenate(train_srcs, axis=0)
        train_tgts = np.concatenate(train_tgts, axis=0)
        train_tgts_y = np.concatenate(train_tgts_y, axis=0)
        
        return train_srcs, train_tgts, train_tgts_y
    
    def _create_samples_from_dict(self, network, GT_dict, S_dict, path_idxs):
        """Create samples from dictionary"""
        path_srcs = []
        path_tgts_y = []
        
        for path_i in path_idxs:
            path_src = self.encode_discontinuous_path(network, S_dict, path_i)
            path_srcs.append(path_src)
            
            path_tgt_y = self.create_sequence_target(GT_dict, path_i)
            path_tgts_y.append(path_tgt_y)
        
        path_srcs = np.array(path_srcs)
        path_tgts_y = np.array(path_tgts_y)
        path_tgts = path_tgts_y[:, :-1]  # Note: 目标输入：去掉最后一个
        path_tgts_y = path_tgts_y[:, 1:]  # Note: 目标标签：去掉第一个
        
        return path_srcs, path_tgts, path_tgts_y


class RoutesFormer:
    """
    RoutesFormer class
    
    Provides full path inference functionality, including training and prediction
    """
    def __init__(self, network, experiment_config, model_config, method_config):
        """
        Initialize RoutesFormer
        
        Args:
            network: Road network object
            experiment_config: Experiment configuration
            model_config: Model configuration
            method_config: Method configuration
        """
        self.max_len = experiment_config.max_len
        self.attributes_dict = experiment_config.attributes_dict
        
        # Special token
        self.tokens = ['bos', 'eos', 'mos', 'pos']
        self.token_indexs = {}
        for i, token in enumerate(self.tokens):
            self.token_indexs[token] = get_num_links(network) + i
        
        # Method configuration
        self.path_generation_method = method_config.path_generation_method
        self.graph_constraint = method_config.graph_constraint
        self.discontinuous_path_attention = method_config.discontinuous_path_attention
        self.allow_decoupled_trying = method_config.allow_decoupled_trying
        self.use_shortest_path = False
        
        self.verbose = False
        
        self.data_generator = DataGenerator(
            network, experiment_config, model_config, method_config
        )
        
        self.model = RoutesFormerTransformer(
            model_config,
            self.attributes_dict,
            self.token_indexs
        )
    
    def train(self, network, GT_dict, path_idxs, use_iterative=False, 
              data_regen_interval=100, num_iterations=5):
        """
        Train model
        
        Args:
            network: Road network object
            GT_dict: Ground truth path dictionary
            path_idxs: Training path indices
            use_iterative: Whether to use iterative training (re-generate data periodically)
            data_regen_interval: How many epochs to regenerate data
            num_iterations: Total number of iterations
        """
        logger.info("Start training RoutesFormer...")
        
        if not use_iterative:
            # Standard training mode: generate all data at once
            logger.info("Use standard training mode (generate all data at once)")
            train_srcs, train_tgts, train_tgts_y = self.data_generator.create_dataset_once(
                network, GT_dict, path_idxs
            )
            self.model.train(train_srcs, train_tgts, train_tgts_y)
        else:
            # Iterative training mode: re-generate data periodically
            logger.info("Use iterative training mode (data augmentation)")
            logger.info(f"  - Regenerate data every {data_regen_interval} epochs")
            logger.info(f"  - Total {num_iterations} iterations")
            logger.info(f"  - Total training epochs: {data_regen_interval * num_iterations}")
            
            # Save original epoch_num, temporarily modify to the number of epochs per iteration
            original_epoch_num = self.model.epoch_num
            self.model.epoch_num = data_regen_interval
            
            for iteration in range(num_iterations):
                logger.info("\n" + "="*60)
                logger.info(f"Iteration {iteration + 1}/{num_iterations}: Regenerate training data")
                logger.info("="*60)
                
                # Regenerate training data (random masking)
                train_srcs, train_tgts, train_tgts_y = self.data_generator.create_dataset_once(
                    network, GT_dict, path_idxs
                )
                
                # Continue training (except for the first iteration, all others are continue training)
                continue_training = (iteration > 0)
                start_epoch = iteration * data_regen_interval
                
                self.model.train(
                    train_srcs, train_tgts, train_tgts_y,
                    continue_training=continue_training,
                    start_epoch=start_epoch
                )
                
                # Release memory
                torch.cuda.empty_cache()
                
                logger.info(f"Iteration {iteration + 1}/{num_iterations} completed")
            
            # Restore original epoch_num
            self.model.epoch_num = original_epoch_num
        
        torch.cuda.empty_cache()
        logger.info("Training completed")
    
    def predict_path(self, network, S_dict, path_i):
        """
        Predict complete path
        
        Args:
            network: Road network object
            S_dict: Sparse observations dictionary
            path_i: Path index
        
        Returns:
            Predicted path and probability
        """
        if self.path_generation_method == 'argmax':
            return self._predict_path_argmax(network, S_dict, path_i)
        else:
            logger.warning(f"Unsupported path generation method: {self.path_generation_method}")
            return {}
    
    def _predict_path_argmax(self, network, S_dict, path_i):
        """
        Use argmax strategy to predict path (greedy decoding)
        
        Args:
            network: Road network object
            S_dict: Sparse observations dictionary
            path_i: Path index
        
        Returns:
            Predicted path dictionary, format: {path_tuple: probability}
        """
        neighbor_links_O = network.graph['neighbor_links_O']
        S_path = tuple(list(S_dict['paths'][path_i]))
        
        path_probability = {}
        
        # Encode input
        src = self.data_generator.encode_discontinuous_path(network, S_dict, path_i)
        src = torch.from_numpy(src).float()
        tgt = torch.LongTensor([[self.token_indexs['bos']]])
        
        # Track visited observed link indices
        s_path_idx = 0
        
        # Autoregressive generation
        for current_step in range(self.max_len):
            # Predict next link
            predict_val = self.model.predict(src, tgt)[-1, 0, :]
            
            if tgt[0][-1] == self.token_indexs['bos']:
                # At the beginning, select start point (fixed use start point of observed path)
                possible_neighbor_link = S_path[0]
                s_path_idx = 1  # Already used first observed point
            else:
                # Select next link based on graph constraint
                current_link = int(tgt[0][-1])
                
                # Get possible adjacent links
                possible_neighbor_links = neighbor_links_O.get(current_link, []).copy()
                
                # Check if current path contains all observed links
                current_path_list = list(tgt.cpu().detach().numpy().ravel())
                all_obs_included = is_path_subsequence_of_path(S_path, current_path_list)
                
                # If there are still unobserved observed links, prioritize next observed link
                if s_path_idx < len(S_path) and not all_obs_included:
                    next_obs_link = S_path[s_path_idx]
                    # If next observed link is in possible adjacent links, force select it
                    if next_obs_link in possible_neighbor_links:
                        possible_neighbor_link = next_obs_link
                        s_path_idx += 1
                    else:
                        # Do not use shortest path during prediction, directly predict from model
                        if len(possible_neighbor_links) <= 0:
                            break
                        link_probabilities = predict_val[possible_neighbor_links]
                        if torch.max(link_probabilities) <= 0:
                            break
                        possible_neighbor_link = possible_neighbor_links[torch.argmax(link_probabilities)]
                elif all_obs_included:
                    # All observed links are included, can end
                    possible_neighbor_links = [self.token_indexs['eos']]
                    possible_neighbor_link = self.token_indexs['eos']
                else:
                    # Normal use model prediction
                    if len(possible_neighbor_links) <= 0:
                        break
                    link_probabilities = predict_val[possible_neighbor_links]
                    if torch.max(link_probabilities) <= 0:
                        break
                    possible_neighbor_link = possible_neighbor_links[torch.argmax(link_probabilities)]
            
            # End condition
            if possible_neighbor_link == self.token_indexs['eos']:
                break
            
            # Add to sequence
            y = torch.LongTensor([[possible_neighbor_link]])
            tgt = torch.cat([tgt, y], dim=1)
        
        # Extract path
        tgt = list(tgt.cpu().detach().numpy().ravel())
        if self.token_indexs['bos'] in tgt:
            tgt.remove(self.token_indexs['bos'])
        
        # Validate path - check if model prediction is successful
        if is_path_subsequence_of_path(S_path, tuple(tgt)):
            # Model prediction successful, use model output
            path_probability[tuple(tgt)] = 1.0
        elif self.use_shortest_path:
            # Model prediction failed, use shortest path as fallback
            try:
                import networkx as nx
                link_nodes_dict = network.graph['link_nodes_dict']
                
                # Generate complete path using shortest paths between observed links
                complete_path = []
                for i in range(len(S_path)):
                    if i == 0:
                        # Add first observed link
                        complete_path.append(S_path[0])
                    elif i < len(S_path):
                        # Connect from previous observed link to current observed link
                        prev_link = S_path[i-1]
                        curr_link = S_path[i]
                        
                        # Get end node of previous link
                        prev_end_node = link_nodes_dict[prev_link][1]
                        # Get start node of current link
                        curr_start_node = link_nodes_dict[curr_link][0]
                        
                        # Calculate shortest path if not the same node
                        if prev_end_node != curr_start_node:
                            try:
                                shortest_path_nodes = nx.shortest_path(network, prev_end_node, curr_start_node)
                                # Convert node path to link path
                                for j in range(len(shortest_path_nodes) - 1):
                                    from_node = shortest_path_nodes[j]
                                    to_node = shortest_path_nodes[j + 1]
                                    # Find corresponding link
                                    for link, (n1, n2) in link_nodes_dict.items():
                                        if n1 == from_node and n2 == to_node:
                                            complete_path.append(link)
                                            break
                            except:
                                # If shortest path fails, just add the gap link directly
                                pass
                        
                        # Add current observed link
                        complete_path.append(curr_link)
                
                # Validate the shortest path fallback result
                if complete_path and is_path_subsequence_of_path(S_path, tuple(complete_path)):
                    path_probability[tuple(complete_path)] = 1.0
                    logger.debug(f"Path {path_i}: Model prediction failed, using shortest path fallback")
            except Exception as e:
                logger.debug(f"Path {path_i}: Shortest path fallback failed: {str(e)}")
        
        return path_probability

