import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import numpy as np
import networkx as nx
import torch_geometric.datasets as pyg_datasets
import matplotlib.pyplot as plt
from torch_geometric.utils import *
from process import full_load_data
import community as community_louvain


# Assuming G is your graph
G = nx.karate_club_graph()  # Example graph, replace with your own

# Perform community detection using the Louvain method
partition = community_louvain.best_partition(G)

# Function to compute layout for each subgraph and adjust positions
def compute_cluster_layouts(graph, partition):
    """
    Compute layouts for each cluster and adjust to avoid overlap.
    """
    pos_total = {}
    cluster_centers = {}
    clusters = set(partition.values())

    # Define initial cluster center positions on a circle
    radius = 10  # Radius of the circle on which to place cluster centers
    angles = np.linspace(0, 2 * np.pi, len(clusters), endpoint=False)
    cluster_positions = np.column_stack([np.cos(angles), np.sin(angles)]) * radius

    for i, cluster in enumerate(clusters):
        # Extract nodes belonging to the current cluster
        nodes = [node for node, cls in partition.items() if cls == cluster]
        subgraph = graph.subgraph(nodes)

        # Compute spring layout for the subgraph
        pos_subgraph = nx.spring_layout(subgraph)

        # Adjust positions based on the cluster center
        center = cluster_positions[i]
        for node, pos in pos_subgraph.items():
            pos_total[node] = pos + center
        cluster_centers[cluster] = center

    return pos_total, cluster_centers

# Compute adjusted layout
pos, cluster_centers = compute_cluster_layouts(G, partition)

# Plotting
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=list(partition.values()), cmap='viridis', alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.2)
for cluster, center in cluster_centers.items():
    plt.text(center[0], center[1], f'Cluster {cluster}', fontsize=15, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6))

plt.title('Graph Clustering with Separated Clusters')
plt.axis('off')
plt.show()
