import networkx as nx
import dgl

import torch as th
from utils.inits import reset, init_weights
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class sLayer(nn.Module):
    def __init__(self,
                 g,
                 node_dim,
                 edge_dim,
                 out_feats,
                 activation=None,
                 dropout=None,
                 bias=None,
                 is_cuda=True):
        super(sLayer, self).__init__()

        # 1. set parameters
        self.g = g
        self.node_dim = node_dim
        self.out_feats = out_feats
        self.activation = activation
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.bias = bias
        self.is_cuda = is_cuda

        # 2. create variables
        self._build_parameters()

        # 3. initialize variables
        self.apply(init_weights)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.linear)

    def _build_parameters(self):
        if self.edge_dim is not None:
            self.e_group_attn_fc = nn.Linear(self.edge_dim, 1, bias=False)
        
        self.linear = nn.Linear(self.edge_dim, self.out_feats, bias=self.bias)

        # Dropout module
        if self.dropout:
            self.dropout = nn.Dropout(p=self.dropout)
    
    def edge_weight_same(self, edges):
        ''' weigh edges from same src and dst '''

        print("\t*** edges.src", edges.src)


    def edge_weight(self, edges):
        print("edges.data['e'] ", edges.data['e'])
        print("edges.src['nid'] ", edges.src['nid'])
        print("edges.dst['nid'] ", edges.dst['nid'])
        print("\tedges.dst['nid'] shape", edges.dst['nid'].shape)
        print("\tedges.data['e'] shape", edges.data['e'].shape)

        # src_list = edges.src['nid'].squeeze(0).type(th.int).tolist()
        # dst_list = edges.dst['nid'].squeeze(0).type(th.int).tolist()
        # print("\tsrc_list", src_list)
        # print("\tdst_list", dst_list)
        # edges_ids = self.g.edge_ids(src_list, dst_list)
        # print("\tedges_ids", edges_ids)
        # SG = self.g.edge_subgraph(edges_ids)
        # print('~~SG', SG.edata)

        # SG.group_apply_edges(func=self.edge_weight_same, group_by='dst')

        src_unique_list = np.unique(edges.src['nid'].squeeze(0).type(th.int).numpy())
        dst_unique_list = np.unique(edges.dst['nid'].squeeze(0).type(th.int).numpy())
        print("\tsrc_unique_list", src_unique_list)
        print("\tdst_unique_list", dst_unique_list)


        # e = []

        for src_n in src_unique_list:
            for dst_n in dst_unique_list:
                edges_sd_euv = self.g.edge_ids(src_n, dst_n, force_multi=True)
                print('\t * src_n', src_n)
                print('\t * dst_n', dst_n)
                # print('\t => edges_sd_euv', edges_sd_euv)
                edges_sd = edges_sd_euv[2] # id of edges from eu[i] to ev[i]
                print('\t => edges_sd', edges_sd)
                
                e = edges_sd
        
                a = self.e_group_attn_fc(e)
                alpha = F.softmax(a, dim=1)
                e_tot = e.shape[1]
                print('e_tot', e_tot)
                print('=> e/e_tot', e/e_tot)
                print('alpha', alpha)
                print('sum alpha', th.sum(alpha))

        # print("\t *** edges.data['e'] ", edges.data['e'])
        # print("\t *** edges.src['nid'] ", edges.src['nid'])
        # print("\t *** edges.dst['nid'] ", edges.dst['nid'])
        # print("\t\t *** edges.dst['nid'] shape", edges.dst['nid'].shape)
        # print("\t\t *** edges.data['e'] shape", edges.data['e'].shape)

        # e_filtered = g.filter_edges(has_dst_one)
        # print('e_filtered: ', e_filtered)

        # e = edges.data['e']#.squeeze(0)
        # print('=> e', e)
        # a = self.e_group_attn_fc(e)
        # alpha = F.softmax(a, dim=1)
        # e_tot = e.shape[1]
        # print('e_tot', e_tot)
        # print('=> e/e_tot', e/e_tot)
        # print('alpha', alpha)
        # print('sum alpha', th.sum(alpha))

        # e = e/e_tot
        # e_w = alpha * e
        # print('=> e_w', e_w)
        # print('-------------')
        # return {'e_w': e_w}

        # # e_filtered = g.filter_edges(has_dst_one)
        # # print('e_filtered: ', e_filtered)

        # e = edges.data['e']#.squeeze(0)
        # print('=> e', e)
        # a = self.e_group_attn_fc(e)
        # alpha = F.softmax(a, dim=1)
        # e_tot = e.shape[1]
        # print('e_tot', e_tot)
        # print('=> e/e_tot', e/e_tot)
        # print('alpha', alpha)
        # print('sum alpha', th.sum(alpha))

        # e = e/e_tot
        # e_w = alpha * e
        # print('=> e_w', e_w)
        # print('-------------')
        # return {'e_w': e_w}

    def forward(self, node_features, edge_features, g):
        if g is not None:
            self.g = g

        # 1. clean graph features
        # reset_graph_features(self.g)

        # 2. set current iteration features
        self.g.ndata['z'] = node_features
        self.g.edata['e'] = edge_features

        # 3. aggregate messages
        if self.edge_dim is not None:
            self.g.group_apply_edges(func=self.edge_weight, group_by='src')

        # self.g.update_all(self.gnn_msg,
        #                   self.gnn_reduce,
        #                   self.node_update)
        # print('-------------------------------')

        h = self.g.ndata.pop('z')
        return h


def msg_fcn(edges):
    print('msg_fcn~~~~~~')
    print("\t edges.data['e']", edges.data['e'])
    print("\t edges.src['z']", edges.src['z'])
    print("\t edges.dst['z']", edges.dst['z'])
    print('\n')
    return {'zsrc': edges.src['z'], 'zdst': edges.dst['z'], 'e': edges.data['e']}

def reduce_fcn(nodes):
    print("nodes.mailbox['zsrc']", nodes.mailbox['zsrc'])
    print("nodes.mailbox['zdst']", nodes.mailbox['zdst'])
    print("nodes.mailbox['e']", nodes.mailbox['e'])
    sum_edges = th.sum(nodes.mailbox['e'])
    new_z = nodes.mailbox['zsrc'] + sum_edges
    # print("nodes.mailbox['zsrc']", nodes.mailbox['zsrc'])
    print('\n')

    return {'z': new_z}


def e_fcn(edges):
    print('e_fcn~~~~~~')
    print("\t edges.data['e']", edges.data['e'])
    print("\t edges.src['z']", edges.src['z'])
    print("\t edges.dst['z']", edges.dst['z'])
    return {'e': edges.src['z']+edges.data['e']}


def edge_weight(edges):
    print("edges.data['e'] ", edges.data['e'])
    print("edges.src['nid'] ", edges.src['nid'])
    print("edges.dst['nid'] ", edges.dst['nid'])
    print("\tedges.dst['nid'] shape", edges.dst['nid'].shape)
    print("\tedges.data['e'] shape", edges.data['e'].shape)

    # e_filtered = g.filter_edges(has_dst_one)
    # print('e_filtered: ', e_filtered)

    e = edges.data['e']#.squeeze(0)
    print('=> e', e)
    e_sum = th.sum(edges.data['e'])
    print('e_sum', e_sum)
    print('-------------')
    # return {'e_w': e/e_sum}
    return {'e_w': e}


def has_dst_one(edges):
    dst_tensor = th.tensor([0.,0.])

    dst_nodes = edges.dst['z']
    print("dst_nodes (edges.dst['z'])", dst_nodes)

    diff = dst_nodes - dst_tensor
    print('diff', diff)

    diff_sum = abs(diff).sum(-1)
    print('diff_sum', diff_sum)

    loc = th.where(diff_sum==0)
    e_filtered = edges.data['e'][loc]
    print("~~ edges.data['e']", edges.data['e'])
    print('~~ e_filtered', e_filtered)

    return e_filtered

    # print("edges.dst['z']===", edges.dst['z'])
    # return (edges.dst['z'] == th.Tensor([ [1, -1]] ))
    # return (edges.dst['z'] == 1 )



g = dgl.DGLGraph(multigraph=True)
for i in range(0,3):
    g.add_nodes(1, data={'z': th.Tensor([ [i, -i]] )})
# g.add_nodes(5)
# A couple edges one-by-one
# for i in range(1, 5):
#     g.add_edge(i, 0, data={'e': th.Tensor([1])})

# A few more with a paired list
# src = list(range(1, 5))
# dst = [0]*len(src)
# g.add_edges(src, dst, data={'e': th.Tensor([1])})

# finish with a pair of tensors
# src = th.tensor([1, 4])
# dst = th.tensor([0, 0])
# g.add_edges(src, dst, data={'e': th.Tensor([1])})
g.add_edge(2, 1, data={'e': th.Tensor([ [3,-3] ]), 't': th.Tensor([0])})
g.add_edge(1, 0, data={'e': th.Tensor([ [1,-1] ]), 't': th.Tensor([1])})
g.add_edge(1, 0, data={'e': th.Tensor([ [2,-2] ]), 't': th.Tensor([2])})
g.add_edge(2, 0, data={'e': th.Tensor([ [4,-4] ]), 't': th.Tensor([0])})
g.add_edge(2, 0, data={'e': th.Tensor([ [5,-5] ]), 't': th.Tensor([1])})
# g.add_edge(4, 0, data={'e': th.Tensor([2])})

nodes_num = g.number_of_nodes()
edges_num = g.number_of_edges()
print('nodes num', nodes_num)
print('edges num', edges_num)

# save nodeid and edgeid to each node and edge
g.ndata['nid'] = th.zeros(nodes_num)
g.edata['eid'] = th.zeros(edges_num)

for nid in range(nodes_num):
    g.ndata['nid'][nid] = th.tensor([nid])
for eid in range(edges_num):
    g.edata['eid'][eid] = th.tensor([eid])


print(g.edata['e'].shape)
# edges = g.edges()

print('\n*** Init val ***')
for i in range(nodes_num):
    for s in range(nodes_num):
        eids = g.edge_id(i, s)
        if len(eids) > 0:
            edatas = []
            for eid in eids:
                edatas.append(g.edata['e'][eid])
                print('eid', eid, 'edata[eid]', g.edata['eid'][eid])
            print(i, '->', s, ' | ', len(eids), 'edges: ', eids)
            print('\t', i, g.ndata['z'][i], '    ->    ', s, g.ndata['z'][s])
            print('\t edatas', edatas)
print('\n')

# g.edata['e_w'] = th.zeros((g.number_of_edges(), 1))
# g.apply_edges(e_fcn)
# g.group_apply_edges(func=edge_weight, group_by='src') # Apply func to the first edge.




class Model(nn.Module):

    type_weight = 5.0

    edge_features_use = 'all'
    node_features_use = 'all'

    def __init__(self, g, is_cuda=True, batch_size=1):
        super(Model, self).__init__()

        self.is_cuda = is_cuda
        self.g = g
        self.batch_size = batch_size

        self.out_feats = 3
        self.n_classes = 2

        self.build_model()


    def build_model(self):

        self.edgnn_layers = nn.ModuleList()
        edGNN = sLayer(g, 1, 2, self.out_feats, F.relu)
        self.edgnn_layers.append(edGNN)

        self.fc = nn.Linear(self.out_feats, self.n_classes)

    def forward(self, g):
        if g is not None:
            g.set_n_initializer(dgl.init.zero_initializer)
            g.set_e_initializer(dgl.init.zero_initializer)
        self.g = g

        node_features = self.g.ndata['z']
        edge_features = self.g.edata['e']

        for layer_idx, layer in enumerate(self.edgnn_layers):
            if layer_idx == 0:  # these are gat layers
                h = node_features

            h = layer(h, edge_features, self.g)
            # save only last layer output
            if layer_idx == len(self.edgnn_layers)-1:
                key = 'h_' + str(layer_idx)
                self.g.ndata[key] = h


graphs = [g]
batches = dgl.batch(graphs)
        
model = Model(g=g)
model.eval()
with th.no_grad():
    logits = model(batches)
    print('~~logits', logits)



# print('\n*** Update edge ***')
# for i in range(nodes_num):
#     for s in range(nodes_num):
#         eids = g.edge_id(i, s)
#         if len(eids) > 0:
#             edatas = []
#             edatas_w = []
#             for eid in eids:
#                 edatas.append(g.edata['e'][eid])
#                 edatas_w.append(g.edata['e_w'][eid])
#             print(i, '->', s, eids)
#             print('\t', i, g.ndata['z'][i], '    ->    ', s, g.ndata['z'][s])
#             print('\t edatas', edatas, '    |    edatas_w', edatas_w)
# print('\n')



# g.update_all(message_func=msg_fcn, reduce_func=reduce_fcn)

# print('\n*** Update node ***')
# for i in range(0,3):
#     print(i, g.ndata['z'][i])
# print('\n')


# Edge broadcasting will do star graph in one go!
# g.clear()
# g.add_nodes(3)

# src = th.tensor(list(range(1, 3)))
# g.add_edges(src, 0)


# import networkx as nx
# import matplotlib.pyplot as plt
# nx.draw(g.to_networkx(), with_labels=True)
# plt.show()