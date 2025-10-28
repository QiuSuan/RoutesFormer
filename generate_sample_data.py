"""
Generate Simulated Data Script

Used to generate simulated grid road network and random path data for quick testing of RoutesFormer model.

Generated data includes:
- road_network.gml: Road network file (NetworkX format)
- neighbor_links_O.npy: Downstream adjacency relationships of links
- neighbor_links_D.npy: Upstream adjacency relationships of links
- GT_dict.npy: Ground truth path data
- network_visualization.png: Network visualization

All files will be saved to the data/ directory.

Usage:
    python generate_sample_data.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# ==================== Configuration Parameters ====================
# Road network parameters
M = 10  # Grid rows
N = 10  # Grid columns
PERTURBATION = 0.15  # Node position perturbation (0-0.5)

# Path generation parameters
NUM_PATHS = 300  # Number of generated paths (>200)
MIN_PATH_LENGTH = 8  # Minimum path length
MAX_PATH_LENGTH = 25  # Maximum path length

# Output directory
OUTPUT_DIR = 'data'

# Random seed (optional, for reproducible results)
RANDOM_SEED = 42


def create_grid_network(m, n, perturbation=0.1):
    """
    Create M*N grid road network, nodes with random perturbation
    
    Args:
        m: Number of rows
        n: Number of columns
        perturbation: Node position perturbation amplitude
    
    Returns:
        G: NetworkX directed graph
        node_positions: Node position dictionary
    """
    print(f"Generating {m}x{n} grid road network...")
    G = nx.DiGraph()
    
    # Create nodes
    node_positions = {}
    node_id = 0
    for i in range(m):
        for j in range(n):
            # Base position + random perturbation
            x = float(j + np.random.uniform(-perturbation, perturbation))
            y = float(i + np.random.uniform(-perturbation, perturbation))
            
            node_positions[node_id] = (x, y)
            G.add_node(node_id, LON=x, LAT=y)
            node_id += 1
    
    # Create edges (links) - bidirectional roads
    edge_id = 0
    for i in range(m):
        for j in range(n):
            current = i * n + j
            
            # Right edge
            if j < n - 1:
                right = i * n + (j + 1)
                length = float(np.linalg.norm(
                    np.array(node_positions[current]) - np.array(node_positions[right])
                ))
                G.add_edge(current, right, 
                          EDGEID=int(edge_id), 
                          LENGTH=length, 
                          VLENGTH=length,
                          FC=1, DF=1, FW=1, RSTRUCT=0)
                edge_id += 1
            
            # Downward edge
            if i < m - 1:
                down = (i + 1) * n + j
                length = float(np.linalg.norm(
                    np.array(node_positions[current]) - np.array(node_positions[down])
                ))
                G.add_edge(current, down, 
                          EDGEID=int(edge_id), 
                          LENGTH=length, 
                          VLENGTH=length,
                          FC=1, DF=1, FW=1, RSTRUCT=0)
                edge_id += 1
            
            # Left edge
            if j > 0:
                left = i * n + (j - 1)
                length = float(np.linalg.norm(
                    np.array(node_positions[current]) - np.array(node_positions[left])
                ))
                G.add_edge(current, left, 
                          EDGEID=int(edge_id), 
                          LENGTH=length, 
                          VLENGTH=length,
                          FC=1, DF=1, FW=1, RSTRUCT=0)
                edge_id += 1
            
            # Upward edge
            if i > 0:
                up = (i - 1) * n + j
                length = float(np.linalg.norm(
                    np.array(node_positions[current]) - np.array(node_positions[up])
                ))
                G.add_edge(current, up, 
                          EDGEID=int(edge_id), 
                          LENGTH=length, 
                          VLENGTH=length,
                          FC=1, DF=1, FW=1, RSTRUCT=0)
                edge_id += 1
    
    # Add graph attributes
    G.graph['data_source'] = 'Simulated'
    G.graph['name'] = f'Grid_{m}x{n}'
    
    print(f"  Number of nodes: {len(G.nodes)}")
    print(f"  Number of edges: {len(G.edges)}")
    
    return G, node_positions


def visualize_and_save_network(G, node_positions, save_path):
    """
    Visualize network and save as PNG
    
    Args:
        G: NetworkX graph
        node_positions: Node position dictionary
        save_path: Save path
    """
    print("Generating network visualization...")
    
    plt.figure(figsize=(12, 12))
    
    # Draw edges
    for u, v in G.edges():
        x = [node_positions[u][0], node_positions[v][0]]
        y = [node_positions[u][1], node_positions[v][1]]
        plt.plot(x, y, 'b-', alpha=0.4, linewidth=1.5)
    
    # Draw nodes
    x_coords = [pos[0] for pos in node_positions.values()]
    y_coords = [pos[1] for pos in node_positions.values()]
    plt.scatter(x_coords, y_coords, c='red', s=80, zorder=5, alpha=0.8)
    
    plt.xlabel('X (longitude)', fontsize=12)
    plt.ylabel('Y (latitude)', fontsize=12)
    plt.title(f'Simulated network ({len(G.nodes)} nodes, {len(G.edges)} edges)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Network visualization saved: {save_path}")
    plt.close()


def create_neighbor_links(G):
    """
    Generate link adjacency relationships
    
    Returns:
        neighbor_links_O: Downstream adjacent links {link_id: [neighbor_ids]}
        neighbor_links_D: Upstream adjacent links {link_id: [neighbor_ids]}
    """
    print("Generating neighbor links...")
    
    neighbor_links_O = defaultdict(list)  # Outlet adjacent links
    neighbor_links_D = defaultdict(list)  # Entry adjacent links
    
    # Build node to link mapping
    node_to_outgoing_links = defaultdict(list)
    node_to_incoming_links = defaultdict(list)
    
    for u, v, data in G.edges(data=True):
        link_id = data['EDGEID']
        node_to_outgoing_links[u].append(link_id)
        node_to_incoming_links[v].append(link_id)
    
    # Build link adjacency relationships
    for u, v, data in G.edges(data=True):
        current_link = data['EDGEID']
        
        # Downstream adjacent (from the end of the current link)
        neighbor_links_O[current_link] = node_to_outgoing_links[v]
        
        # Upstream adjacent (to the start of the current link)
        neighbor_links_D[current_link] = node_to_incoming_links[u]
    
    # Convert to regular dictionary
    neighbor_links_O = dict(neighbor_links_O)
    neighbor_links_D = dict(neighbor_links_D)
    
    print(f"  Number of downstream adjacent links: {len(neighbor_links_O)}")
    print(f"  Number of upstream adjacent links: {len(neighbor_links_D)}")
    
    return neighbor_links_O, neighbor_links_D


def generate_random_path(G, neighbor_links_O, min_length=8, max_length=25, max_attempts=100):
    """
    Generate a random path in the road network
    
    Args:
        G: Road network graph
        neighbor_links_O: Downstream adjacent links
        min_length: Minimum path length
        max_length: Maximum path length
        max_attempts: Maximum number of attempts
    
    Returns:
        path: Path (list of link IDs)
    """
    all_links = list(G.edges(data=True))
    
    for _ in range(max_attempts):
        # Randomly select starting link
        start_edge = np.random.choice(len(all_links))
        u, v, data = all_links[start_edge]
        current_link = data['EDGEID']
        
        path = [current_link]
        
        # Target path length
        target_length = np.random.randint(min_length, max_length + 1)
        
        # Generate path
        while len(path) < target_length:
            # Get downstream adjacent links
            neighbors = neighbor_links_O.get(current_link, [])
            
            if not neighbors:
                break
            
            # Filter visited links (avoid short loops)
            available_neighbors = [n for n in neighbors if n not in path[-min(3, len(path)):]]
            
            if not available_neighbors:
                available_neighbors = neighbors
            
            # Randomly select next link
            next_link = np.random.choice(available_neighbors)
            path.append(next_link)
            current_link = next_link
        
        # Check if path length meets requirements
        if len(path) >= min_length:
            return path
    
    # If failed, return a minimum length path
    return path[:max(min_length, len(path))]


def generate_path_dataset(G, neighbor_links_O, num_paths, min_length, max_length):
    """
    Generate path dataset
    
    Returns:
        GT_dict: Ground truth path dictionary
    """
    print(f"Generating {num_paths} random paths...")
    paths = {}
    
    for i in range(num_paths):
        path = generate_random_path(G, neighbor_links_O, min_length, max_length)
        paths[i] = np.array(path)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_paths} paths")
    
    GT_dict = {
        'paths': paths
    }
    
    # Count path lengths
    path_lengths = [len(path) for path in GT_dict['paths'].values()]
    print(f"  Average path length: {np.mean(path_lengths):.2f}")
    print(f"  Path length range: {np.min(path_lengths)} - {np.max(path_lengths)}")
    
    return GT_dict


def main():
    """Main function"""
    print("="*60)
    print("RoutesFormer simulated data generation script")
    print("="*60)
    
    # Set random seed
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        print(f"Random seed: {RANDOM_SEED}")
    
    print(f"\nConfiguration parameters:")
    print(f"  Network size: {M}x{N}")
    print(f"  Node perturbation: {PERTURBATION}")
    print(f"  Number of paths: {NUM_PATHS}")
    print(f"  Path length: {MIN_PATH_LENGTH}-{MAX_PATH_LENGTH}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Generate network
    network, node_positions = create_grid_network(M, N, PERTURBATION)
    
    # 2. Visualize network
    viz_path = os.path.join(OUTPUT_DIR, 'network_visualization.png')
    visualize_and_save_network(network, node_positions, viz_path)
    
    # 3. Generate neighbor links
    neighbor_links_O, neighbor_links_D = create_neighbor_links(network)
    
    # 4. Generate path data
    GT_dict = generate_path_dataset(
        network, 
        neighbor_links_O, 
        NUM_PATHS, 
        MIN_PATH_LENGTH, 
        MAX_PATH_LENGTH
    )
    
    # 5. Save all files
    print("\nSaving data files...")
    
    # Save network
    network_file = os.path.join(OUTPUT_DIR, 'road_network.gml')
    nx.write_gml(network, network_file)
    print(f"  ✓ {network_file}")
    
    # Save neighbor links
    neighbor_o_file = os.path.join(OUTPUT_DIR, 'neighbor_links_O.npy')
    np.save(neighbor_o_file, neighbor_links_O)
    print(f"  ✓ {neighbor_o_file}")
    
    neighbor_d_file = os.path.join(OUTPUT_DIR, 'neighbor_links_D.npy')
    np.save(neighbor_d_file, neighbor_links_D)
    print(f"  ✓ {neighbor_d_file}")
    
    # Save path data
    gt_dict_file = os.path.join(OUTPUT_DIR, 'GT_dict.npy')
    np.save(gt_dict_file, GT_dict)
    print(f"  ✓ {gt_dict_file}")
    
    # Summary
    print("\n" + "="*60)
    print("Data generation completed!")
    print("="*60)
    print(f"\nGenerated files located in '{OUTPUT_DIR}/' directory:")
    print(f"  - road_network.gml ({len(network.nodes)} nodes, {len(network.edges)} edges)")
    print(f"  - neighbor_links_O.npy ({len(neighbor_links_O)} links)")
    print(f"  - neighbor_links_D.npy ({len(neighbor_links_D)} links)")
    print(f"  - GT_dict.npy ({len(GT_dict['paths'])} paths)")
    print(f"  - network_visualization.png (network visualization)")
    print(f"\nNext steps:")
    print(f"  1. Check {viz_path} to confirm network generation")
    print(f"  2. Run 'python train.py' to start training")
    print(f"  3. Run 'python test.py' to test")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nData generation interrupted by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()

