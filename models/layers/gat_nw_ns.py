import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY, GNN_EDGE_LABELS_KEY, GNN_EDGE_TYPES_KEY, GNN_NODE_TYPES_KEY, GNN_NODE_LABELS_KEY


class GATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_dim, edge_ft_out_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g

        # edge_lbl_dim = self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]

        self.node_lv = NodeLevelGAT(g, node_dim, edge_dim, edge_ft_out_dim, out_dim)


    def forward(self, node_features, edges_features, g):
        if g is not None:
            self.g = g

        z_final = self.node_lv(node_features, edges_features, g)

        return z_final



class NodeLevelGAT(nn.Module):
    """
    Weight by neighbor nodes
    """
    def __init__(self, g, node_in_dim, edge_in_dim, edge_ft_out_dim, out_dim):
        super(NodeLevelGAT, self).__init__()
        self.g = g

        node_out_dim = node_in_dim//2

        # equation (1)
        self.fc_n = nn.Linear(node_in_dim, out_dim, bias=False)
        self.fc_e = nn.Linear(edge_in_dim, edge_ft_out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(out_dim + edge_ft_out_dim  +  out_dim, 1, bias=False)
        # outdim
        self.fc2 = nn.Linear(out_dim + edge_ft_out_dim, out_dim, bias=False)


        # self.attn_fc = nn.Linear(node_in_dim + edge_in_dim  +  node_in_dim, 1, bias=False)
        # self.fc2 = nn.Linear(node_in_dim + edge_in_dim, out_dim, bias=False)

        self.nodes_updated = 0

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        src_node_n_edge_ft = torch.cat((edges.src['z'], edges.data['e_ft']), dim=1)
        # src_node_n_edge_ft__fc = self.fc(src_node_n_edge_ft)
        src_node_n_edge_ft__fc = src_node_n_edge_ft
        # print('src_node_n_edge_ft', src_node_n_edge_ft.shape)
        
        z2 = torch.cat([src_node_n_edge_ft__fc, edges.dst['z']], dim=1)
        # print("edges.dst['z']", edges.dst['z'].shape)
        # print("edges.src['z']", edges.src['z'].shape)
        # print("edges.data['e_ft']", edges.data['e_ft'].shape)
        # print('z2', z2.shape)
        a = self.attn_fc(z2)
        return {'zne': src_node_n_edge_ft__fc, 'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        # print("edges.data['e']", edges.data['e'])
        return {'z': edges.data['zne'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # Equation (3)
        #   Normalize the attention scores using softmax (equation (3)).
        # print("nodes.mailbox['e']", nodes.mailbox['e'])
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # print('alpha', alpha)
        # print('mailbox z', nodes.mailbox['z'].shape)
        # print('alpha shape', alpha.shape)

        # Equation (4)
        #   Aggregate neighbor embeddings weighted by the attention scores (equation(4)).
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        h = self.fc2(h)

        # print('h', h.shape)

        self.nodes_updated += 1

        return {'z': h}


    def forward(self, h, edges_features, g):
        if g is not None:
            self.g = g

        # node-level
        # Equation (1)
        # print('h first', h.shape)
        h = self.fc_n(h)
        edges_features = self.fc_e(edges_features)
        self.g.ndata['z'] = h
        self.g.edata['e_ft'] = edges_features

        # Equation (2)
        # The un-normalized attention score eij is calculated using the embeddings of adjacent nodes i and j. This suggests that the attention scores can be viewed as edge data, which can be calculated by the apply_edges API. The argument to the apply_edges is an Edge UDF, which is defined as below:
        self.g.apply_edges(self.edge_attention)

        # Equation (3) & (4)
        # `update_all` API is used to trigger message passing on all the nodes. The message function sends out two tensors: 
        #   - the transformed z embedding of the source node, and
        #   - the un-normalized attention score e on each edge.
        # The reduce function then performs two tasks:
        #   - Normalize the attention scores using softmax (equation (3)).
        #   - Aggregate neighbor embeddings weighted by the attention scores (equation(4)).
        self.g.update_all(self.message_func, self.reduce_func)

        # print('nodes_updated', self.nodes_updated)

        return self.g.ndata.pop('z')




class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_dim, edge_ft_out_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, node_dim, edge_dim, edge_ft_out_dim, out_dim))
        self.merge = merge

        print('self.heads~~~', self.heads)

    def forward(self, node_features, edges_features, g):
        # head_outs = []
        # for attn_head in self.heads:
        #     head_out = attn_head(node_features, g)
        #     head_outs.append(head_out)
        #     print('head_out~~~', head_out)
        
        head_outs = [attn_head(node_features, edges_features, g) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))
