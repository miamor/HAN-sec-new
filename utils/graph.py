import numpy as np


class Graph(object):
    def __init__(self, g, node_tags, label):
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label

        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()

        ## Add neighbor info
        self.neighbor1 = []
        self.neighbor1_tag = []
        self.neighbor2 = []
        self.neighbor2_tag = []

        for i in range(self.num_nodes):
            self.neighbor1.append(g.neighbors(i))
            self.neighbor1_tag.append([node_tags[w] for w in g.neighbors(i)])
        for i in range(self.num_nodes):
            tmp = []
            for j in self.neighbor1[i]:
                for k in g.neighbors(j):
                    if k != i:
                        tmp.append(k)
            self.neighbor2.append(tmp)
            self.neighbor2_tag.append([node_tags[w] for w in tmp])

