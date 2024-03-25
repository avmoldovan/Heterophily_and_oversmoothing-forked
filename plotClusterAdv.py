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



# List of datasets to load
#datasets = ['Wisconsin', 'Chameleon', 'Cora', 'Squirrel',  'Actor', 'Texas', 'Cornell', 'Pubmed', 'Citeseer']
datasets = ['Cora', 'Chameleon', 'Wisconsin']
#datasets = ['Cora']



fig, axs = plt.subplots(1, 3, figsize=(45, 15), constrained_layout=True)
axs = axs.flatten()  # Flatten to easily iterate


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

for i, dataset_name in enumerate(datasets):

    if dataset_name in ['Pubmed', 'Cora', 'Citeseer']:
        if dataset_name == 'Cora':
            # Load the dataset (this is conceptual, actual loading may vary)
            dataset = pyg_datasets.Planetoid(root='/tmp', name=dataset_name)
            data = dataset[0]  # Assuming we want the first graph in the dataset
            color_map = {label: i for i, label in enumerate(data.y)}
            # Convert to NetworkX for visualization
            G = to_networkx(data, to_undirected=True)
    elif dataset_name in ['Chameleon', 'Squirrel']:
        if dataset_name == 'Chameleon':
            adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels, deg_vec, raw_adj = full_load_data(dataset_name,'splits/chameleon_split_0.6_0.2_0.npz',False, model_type='GGCN', embedding_method='poincare', get_degree=True)
            G = nx.from_numpy_array(adj.detach().cpu().to_dense().numpy())

            color_map = {label: i for i, label in enumerate(labels)}
            #G = to_networkx(features, to_undirected=True)
    elif dataset_name in ['Actor', 'Texas', 'Wisconsin', 'Cornell']:
        if dataset_name == 'Wisconsin':
            adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels, deg_vec, raw_adj = full_load_data(dataset_name,'splits/wisconsin_split_0.6_0.2_0.npz',False, model_type='GGCN', embedding_method='poincare', get_degree=True)
            color_map = {label: i for i, label in enumerate(labels)}
            G = nx.from_numpy_array(adj.detach().cpu().to_dense().numpy())
            #G = to_networkx(features, to_undirected=True)
    # #
    # # # Create a list of colors for each node based on its label
    # # node_colors = [color_map[node_labels[node]] for node in G.nodes()]
    #
    G.remove_edges_from(nx.selfloop_edges(G))

    # Perform community detection using the Louvain method
    partition = community_louvain.best_partition(G)

    # Compute adjusted layout
    pos, cluster_centers = compute_cluster_layouts(G, partition)

    nx.draw_networkx_nodes(G, pos, ax=axs[i], node_size=50, node_color=list(partition.values()), cmap='viridis', alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=axs[i], alpha=0.2)
    # for cluster, center in cluster_centers.items():
    #     axs[i].text(center[0], center[1], f'Cluster {cluster}', fontsize=15, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6))

    axs[i].set_title(dataset_name, fontsize=60)
#plt.title('Graph Clustering with Separated Clusters')
plt.axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
#plt.colorbar()
# plt.savefig('plotCluster.eps',format='eps', dpi=300)
# plt.savefig('plotCluster.pdf',format='pdf', dpi=300)
# plt.savefig('plotCluster.png',format='png', dpi=300)
plt.show()
