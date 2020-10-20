from utils.utils import *
import dgl


G = load_pickle('data_pickle/reverse/TuTu__vocabtutu__iapi__tfidf__topk=3/graphs')
# G = G[13:15]
# G = G[20:22]

# gb = load_pickle('api_tasks/data_graphs/hh/hh__7037__2f9153d8436fbba0955ab38c82bdc61d56bed5e3855e4fbf68dcc678f2da6806__MSWebp_store')
# gg = load_pickle('api_tasks/data_graphs/hh/hh__4925__0b2d51a4d7131a7239909aad1e5f8b9290c527ebe07e7d93886cb1c67d50a5ab__')

gg = G[1]
gb = G[0]

# print('gg', gg)
# print('gb', gb)
# print('gb ndata', gb.ndata)

print('gg hnl', gg.ndata['hnl'].shape)
print('gb hnl', gb.ndata['hnl'].shape)
print('gg hnt', gg.ndata['hnt'].shape)
print('gb hnt', gb.ndata['hnt'].shape)
print('gg number_of_nodes', gg.number_of_nodes())
print('gb number_of_nodes', gb.number_of_nodes())


import matplotlib.pyplot as plt

# pos = nx.spring_layout(gb)
# print('pos', pos)
# nx.draw(gb, pos, with_labels=True)
# nx.draw_networkx_edge_labels(gb, pos)
# plt.show()

# plt.subplot(121)
# nx.draw(gg.to_networkx(), with_labels=True)

# plt.subplot(122)
# nx.draw(gb.to_networkx(), with_labels=True)
# plt.show()


Gbatch = dgl.batch(G)
print('Gbatch', Gbatch)
