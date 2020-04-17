import networkx as nx
import dgl
import matplotlib.pyplot as plt

from utils.utils import *

graphs = load_pickle('/media/fitmta/Storage/MinhTu/HAN_sec_new/data_pickle/reverse/merge__iapi__vocablower_iapi_m__tfidf/graphs')

g = graphs[2]
G = g.to_networkx()
nnum = len(G.nodes)
for i in range(nnum):
    print('nid', g.ndata['nid'][i])
    G.nodes[i]['nid'] = g.ndata['nid'][i]

# enum = len(G.edges)
# for i in range(enum):
#     print('eid', g.edata['eid'][i])
#     G.edges[i]['eid'] = g.edata['eid'][i]

i = 0
print(G.edges)
for (e_fr, e_to, e_id) in G.edges:
    print("g.edges[e_fr, e_to].data['eid']", g.edges[e_fr, e_to].data['eid'])
    G.edges[e_fr, e_to, e_id]['eid'] = g.edges[e_fr, e_to].data['eid']
    i += 1


pos = nx.spring_layout(G)
print('G', G)
print('pos', pos)
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos)
plt.show()

print('-----------------------')

# G = nx.Graph()
# G.add_edge(1, 2, weight=3)
# G.add_edge(2, 3, weight=5)
# pos = nx.spring_layout(G)
# print('G', G)
# print('pos', pos)
# nx.draw(G, pos)
# nx.draw_networkx_edge_labels(G, pos)

# plt.show()