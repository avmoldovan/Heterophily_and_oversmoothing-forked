import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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

import numpy as np
import scipy.sparse as sp



# List of datasets to load
#datasets = ['Wisconsin', 'Chameleon', 'Cora', 'Squirrel',  'Actor', 'Texas', 'Cornell', 'Pubmed', 'Citeseer']
datasets = ['Cora', 'Chameleon', 'Wisconsin']
#datasets = ['Cora']



fig, axs = plt.subplots(1, 3, figsize=(45, 15), constrained_layout=True)
axs = axs.flatten()  # Flatten to easily iterate



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
    G.remove_edges_from(nx.selfloop_edges(G))

    # Position nodes using the spring layout
    pos = nx.spring_layout(G, seed=42)  # For consistent layout between runs

    # Extract labels and assign a unique color to each label
    labels = nx.get_node_attributes(G, 'label')
    unique_labels = set(labels.values())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))


    # Draw nodes with colors according to their label
    for label, color in label_color_map.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n, lbl in labels.items() if lbl == label],
                               node_color=[color], label=label, node_size=100, ax=axs[i])

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=axs[i])
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', ax=axs[i])
    axs[i].set_title(dataset_name, fontsize=60)

plt.subplots_adjust(wspace=0, hspace=0)
# Add legend
#plt.legend(scatterpoints=1, frameon=False, labelspacing=1, bbox_to_anchor=(1.05, 1), loc='upper left')

#plt.title('Graph Visualization with Actual Labels')
#plt.axis('off')  # Turn off the axis
#plt.tight_layout()
plt.show()
