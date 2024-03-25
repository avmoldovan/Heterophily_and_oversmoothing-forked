import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Example: Create a graph with labeled nodes
# In your case, replace this with loading your graph and ensure it has a 'label' attribute for each node
G = nx.Graph()
G.add_nodes_from([
    (1, {'label': 'A'}),
    (2, {'label': 'B'}),
    (3, {'label': 'A'}),
    (4, {'label': 'B'}),
    (5, {'label': 'C'}),
])
G.add_edges_from([(1, 3), (2, 4), (3, 4), (4, 5)])

# Extract labels and assign a unique color to each label
labels = nx.get_node_attributes(G, 'label')
unique_labels = set(labels.values())
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
label_color_map = dict(zip(unique_labels, colors))

# Position nodes using the spring layout
pos = nx.spring_layout(G, seed=42)  # For consistent layout between runs

# Draw nodes with colors according to their label
for label, color in label_color_map.items():
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, lbl in labels.items() if lbl == label],
                           node_color=[color], label=label, node_size=100)

# Draw edges and labels
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

# Add legend
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Graph Visualization with Actual Labels')
plt.axis('off')  # Turn off the axis
plt.tight_layout()
plt.show()
