import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np

# Create a simple igraph Graph
vertices = ["1", "2", "3", "4", "5", "6", "7", "8"]
edges = [(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (3, 5), (6, 7)]
G = ig.Graph(edges=edges, directed=False)
G.vs["label"] = vertices  # Optional: Assigning labels to vertices

# Perform community detection using the Leiden algorithm
partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)

# Get the membership vector indicating the community of each node
membership = partition.membership

# Assign a unique color to each community
unique_communities = set(membership)
colors = plt.cm.get_cmap('viridis', len(unique_communities))
node_colors = [colors(i) for i in membership]

# Plotting
fig, ax = plt.subplots()
# Convert the igraph layout to a format that can be used with matplotlib
layout = G.layout('kk')  # Kamada-Kawai layout for nice spacing
x, y = zip(*[layout[i] for i in range(len(layout))])
scatter = ax.scatter(x, y, c=node_colors, s=100)

# Draw the edges
for edge in edges:
    start, end = layout[edge[0]], layout[edge[1]]
    ax.plot([start[0], end[0]], [start[1], end[1]], c="gray")

# Annotate the nodes with their labels
for i, txt in enumerate(vertices):
    ax.annotate(txt, (x[i], y[i]))

plt.axis('off')  # Turn off the axis
plt.show()
