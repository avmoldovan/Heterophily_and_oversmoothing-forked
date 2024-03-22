import torch_geometric.datasets as pyg_datasets
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import *
from process import full_load_data
import numpy as np

# List of datasets to load
#datasets = ['Wisconsin', 'Chameleon', 'Cora', 'Squirrel',  'Actor', 'Texas', 'Cornell', 'Pubmed', 'Citeseer']
datasets = ['Cora', 'Chameleon', 'Wisconsin']
#datasets = ['Cora']

fig, axs = plt.subplots(1, 3, figsize=(45, 15), constrained_layout=True)
axs = axs.flatten()  # Flatten to easily iterate

node_labels = {0: 'A', 1: 'B', 2: 'A', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
unique_labels = list(set(node_labels.values()))

#plt.colormaps.plasma()

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
    #
    # # Create a list of colors for each node based on its label
    # node_colors = [color_map[node_labels[node]] for node in G.nodes()]

    G.remove_edges_from(nx.selfloop_edges(G))

    max_degree = max(dict(G.degree()).values())
    #node_classes = nx.get_node_attributes(G, 'class')  # Or the correct attribute name

    #node_colors = [unique_labels[node] for node in G.nodes()]
    node_degrees = G.degree
    #max_degree = max(node_degrees.values())
    #node_sizes = [300 * node_degrees[node] / max_degree for node in G.nodes()]  # Scaling

    layout = nx.spring_layout(G, seed=42)  # Seed for some layout consistency
    #layout = nx.kamada_kawai_layout(G)
    cent = nx.degree_centrality(G)
    node_size = list(map(lambda x: x * 500, cent.values()))
    cent_array = np.array(list(cent.values()))
    threshold = sorted(cent_array, reverse=True)[10]
    cent_bin = np.where(cent_array >= threshold, 1, 0.1)

    #
    # # Plotting the graph
    # pos = nx.spring_layout(G)  # Generate a layout to spread out nodes

    nx.draw(G, pos=layout, ax=axs[i],
        #node_color=node_colors,
        node_size=node_size,
        node_color=cent_bin,
        nodelist=list(cent.keys()),
        edge_color='grey',
            linewidths=0.5,
      #      edge_cmap= plt.colormaps.plasma,
        with_labels=False)  # Turn off labels for large graphs
    # # Draw nodes and edges separately to customize colors
    # nx.draw_networkx_nodes(G, ax=axs[i], pos=pos)
    # nx.draw_networkx_edges(G, ax=axs[i], pos=pos, edge_color='gray')  # Customize edge colors here
    # #nx.draw_networkx_labels(G, ax=axs[i], pos=pos)
    # #nx.draw(G, ax=axs[i], with_labels=False, node_size=30)
    axs[i].set_title(dataset_name, fontsize=60)

#plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plot.eps',format='eps', dpi=300)
plt.savefig('plot.pdf',format='pdf', dpi=300)
plt.savefig('plot.png',format='png', dpi=300)
plt.show()
