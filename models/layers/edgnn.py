"""
edGNN layer (add link to the paper)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.inits import reset, init_weights
from utils.utils import reset_graph_features

from utils.constants import *
import networkx as nx


class edGNNLayer(nn.Module):
    """
    Simple GNN layer
    """

    def __init__(self,
                 g,
                 node_dim,
                 edge_dim,
                 out_feats,
                 activation=None,
                 dropout=None,
                 bias=None,
                 is_cuda=True):
        """
        edGNN Layer constructor.

        Args:
            g (dgl.DGLGraph): instance of DGLGraph defining the topology for message passing
            node_dim (int): node dimension
            edge_dim (int): edge dimension (if 1-hot, edge_dim = n_rels)
            out_feats (int): hidden dimension
            activation: pyTorch functional defining the nonlinearity to use
            dropout (float or None): dropout probability
            bias (bool): if True, a bias term will be added before applying the activation
        """
        super(edGNNLayer, self).__init__()

        # 1. set parameters
        self.g = g
        self.node_dim = node_dim
        self.out_feats = out_feats
        self.activation = activation
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.bias = bias
        self.is_cuda = is_cuda
        self.g_viz = None

        # 2. create variables
        self._build_parameters()

        # 3. initialize variables
        self.apply(init_weights)
        self.reset_parameters()

        # self.g_viz = nx.drawing.nx_pydot.read_dot('/media/fitmta/Storage/MinhTu/HAN_sec_new/data/graphviz/cuckoo_ADung/tmalware__0bee54efca5e41b8bcafcba4f70c9498d48e088553f7b442ea54670db9f874b2.json')

    def reset_parameters(self):
        reset(self.linear)

    def _build_parameters(self):
        """
        Build parameters and store them in a dictionary.
        The keys are the same keys of the node features to which we are applying the parameters.
        """
        input_dim = 2 * self.node_dim
        # print('self.node_dim', self.node_dim)
        # print('self.edge_dim', self.edge_dim)
        if self.edge_dim is not None:
            input_dim = input_dim + self.edge_dim
            self.e_group_attn_fc = nn.Linear(input_dim, 1, bias=False)
        # print('input_dim', input_dim)
        # print('self.out_feats', self.out_feats)

        input_dim_attn1 = self.node_dim if self.edge_dim is None else self.node_dim + self.edge_dim
        self.e_src_attn_fc = nn.Linear(input_dim_attn1, 1, bias=False)
        # self.e_src_linear = nn.Linear(input_dim_attn1, self.out_feats, bias=False)

        self.linear = nn.Linear(input_dim, self.out_feats, bias=self.bias)
        # self.attn_fc = nn.Linear(input_dim, 1, bias=False)

        # self.dst_attn_fc = nn.Linear(self.out_feats, 1, bias=False)
        # self.attn_fc = nn.Linear(input_dim, 1, bias=False)

        # Dropout module
        if self.dropout:
            self.dropout = nn.Dropout(p=self.dropout)


    def edge_weight_src(self, edges):
        e_ft = edges.data[GNN_EDGE_FEAT_IN_KEY]
        # src_ft = edges.src[GNN_NODE_FEAT_IN_KEY]
        # print('~~~ edges.src nid', edges.src['nid'])

        if self.edge_dim is not None:
            # z2 = e_ft
            # a = self.e_group_attn_fc(e_ft)
            z2 = torch.cat([e_ft, edges.src[GNN_NODE_FEAT_IN_KEY], edges.dst[GNN_NODE_FEAT_IN_KEY]], dim=2)
            a = self.e_group_attn_fc(z2)
            # e = F.leaky_relu(a)
            e = a
            gamma = F.softmax(e, dim=1)
            e_weighted = gamma * e_ft

            # self.viz(edges, gamma)
        else:
            e_weighted = e_ft

        #####################
        # Inference only
        #####################
        # if self.g_viz is not None:
        #     self.viz_edge(edges, gamma)

        # print('e_ft', e_ft)
        # print('e_weighted', e_weighted)
        # print('gamma', gamma)
        # print('~~~ e_ft', e_ft.shape)
        # print('~~~ e_weighted.shape', e_weighted.shape)

        return {GNN_EDGE_FEAT_OUT_KEY: e_weighted}


    def viz_edge(self, edges, gamma):
        n_src = edges.src['nid']
        n_dst = edges.dst['nid']
        e_ids = edges.data['eid']
        # print('=== src', n_src)
        # print('=== dst', n_dst)
        # print('=== e_ids', e_ids)
        # print('=== src', n_src.shape)
        # print('=== dst', n_dst.shape)
        print('=== e_ids', e_ids.shape)
        gamma_ = gamma.squeeze(2)
        # print('=== gamma_ shape', gamma_.shape)

        for k in range(len(n_src)):
            e_ids_gr = e_ids[k]
            for i in range(len(e_ids_gr)):
                for (e_fr, e_to, e_id) in self.g_viz.edges:
                    edge_data = self.g_viz.edges[e_fr, e_to, e_id]
                    eid = edge_data['eid']

                    # print('eid', eid, 'i', i, 'e_ids_gr[i].item()', e_ids_gr[i].item(), int(eid) == int(e_ids_gr[i].item()))
                    if int(eid) == int(e_ids_gr[i].item()):
                        # print('\t~~~ e_fr', e_fr, 'e_to', e_to, 'e_id', e_id, 'gamma_[k][i]', gamma_[k][i])
                        # print('\t~~~ self.g_viz.edges[e_fr, e_to, e_id]', self.g_viz.edges[e_fr, e_to, e_id])
                        
                        self.g_viz.edges[e_fr, e_to, e_id]['weight'] = gamma_[k][i].item()
                        # print('edge_data', edge_data)
                        if 'label' not in edge_data:
                            edge_data['label'] = ''
                        self.g_viz.edges[e_fr, e_to, e_id]['label'] = '{} ({})'.format(edge_data['label'], round(gamma_[k][i].item(), 2))
                        break


    def gnn_msg(self, edges):
        """
            If edge features: for each edge u->v, return as msg: MLP(concat([h_u, h_uv]))
        """
        if self.g.edata is not None:
            # print('GNN_NODE_FEAT_IN_KEY', GNN_NODE_FEAT_IN_KEY)
            # print('edges.src[GNN_NODE_FEAT_IN_KEY]', edges.src[GNN_NODE_FEAT_IN_KEY])
            # print('edges.data[GNN_EDGE_FEAT_IN_KEY]', edges.data[GNN_EDGE_FEAT_IN_KEY])
            # print('edges.data[GNN_EDGE_FEAT_OUT_KEY]', edges.data[GNN_EDGE_FEAT_OUT_KEY])
            # print('edges.src[GNN_NODE_FEAT_IN_KEY].shape', edges.src[GNN_NODE_FEAT_IN_KEY].shape)
            # print('edges.data[GNN_EDGE_FEAT_OUT_KEY].shape', edges.data[GNN_EDGE_FEAT_OUT_KEY].shape)

            msg = torch.cat([edges.src[GNN_NODE_FEAT_IN_KEY], edges.data[GNN_EDGE_FEAT_OUT_KEY]], dim=1)
            # msg = torch.cat([edges.src[GNN_NODE_FEAT_IN_KEY], edges.data[GNN_EDGE_FEAT_IN_KEY]], dim=1)
            # print('msg', msg)
            # print('msg', msg.shape)
            
            if self.dropout:
                msg = self.dropout(msg)
        else:
            msg = edges.src[GNN_NODE_FEAT_IN_KEY]
            if self.dropout:
                msg = self.dropout(msg)
        # print('edges.src[GNN_NODE_FEAT_IN_KEY]', edges.src[GNN_NODE_FEAT_IN_KEY].shape)
        # print('edges.data[GNN_EDGE_FEAT_OUT_KEY]', edges.data[GNN_EDGE_FEAT_OUT_KEY].shape)
        return {GNN_MSG_KEY: msg}

    def gnn_reduce(self, nodes):
        msg = nodes.mailbox[GNN_MSG_KEY]
        # print('~~~ msg mailbox', msg)
        # print('\t*** nodes nid', nodes.data['nid'])
        # print('\t~~~ msg mailbox', msg.shape)

        a = self.e_src_attn_fc(msg)
        # e = F.elu(a)
        e = a
        alpha = F.softmax(e, dim=1)
        # alpha = e
        msg = alpha * msg

        # msg = self.e_src_linear(msg)
        
        # print('a', a)
        # print('alpha', alpha)
        # print('msg.shape', msg.shape)

        accum = torch.sum((msg), 1)
        # print('nodes.mailbox[GNN_MSG_KEY]', nodes.mailbox[GNN_MSG_KEY])
        return {GNN_AGG_MSG_KEY: accum}

    def node_update(self, nodes):
        # print('nodes', nodes)
        # print('nodes.data[GNN_NODE_FEAT_IN_KEY].shape', nodes.data[GNN_NODE_FEAT_IN_KEY].shape)
        # print('nodes.data[GNN_AGG_MSG_KEY].shape', nodes.data[GNN_AGG_MSG_KEY].shape)
        h = torch.cat([nodes.data[GNN_NODE_FEAT_IN_KEY],
                       nodes.data[GNN_AGG_MSG_KEY]],
                      dim=1)
        # h = h.type(torch.cuda.LongTensor)
        # print('h.shape', h.shape)
        
        h = self.linear(h)
        

        # a = self.dst_attn_fc(h)
        # # e = F.leaky_relu(a)
        # e = a
        # alpha = F.softmax(e, dim=1)

        # self.viz_node(nodes, alpha)


        if self.activation:
            h = self.activation(h)

        if self.dropout:
            h = self.dropout(h)

        # return {GNN_NODE_FEAT_OUT_KEY: h, 'w': alpha}
        return {GNN_NODE_FEAT_OUT_KEY: h}


    def viz_node(self, nodes, alpha):
        n_src = nodes.data['nid']
        gamma_ = alpha.squeeze(2)
        print('=== n_src', n_src)
        print('=== n_src shape', n_src.shape)

        for k in range(len(n_src)):
            for n_id in self.g_viz.nodes:
                print('n_id', n_id)
                node_data = self.g_viz.nodes[n_id]
                nid = node_data['nid']

                print('n_id', n_id, 'k', k, 'nid', nid)




    def forward(self, node_features, edge_features, g):

        if g is not None:
            self.g = g

        # 1. clean graph features
        reset_graph_features(self.g)

        # 2. set current iteration features
        self.g.ndata[GNN_NODE_FEAT_IN_KEY] = node_features
        self.g.edata[GNN_EDGE_FEAT_IN_KEY] = edge_features

        # 3. aggregate messages
        if self.edge_dim is not None:
            self.g.group_apply_edges(func=self.edge_weight_src, group_by='src')

        self.g.update_all(self.gnn_msg,
                          self.gnn_reduce,
                          self.node_update)
        # print('-------------------------------')

        h = self.g.ndata.pop(GNN_NODE_FEAT_OUT_KEY)
        return h
