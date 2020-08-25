import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY, GNN_EDGE_LABELS_KEY, GNN_EDGE_TYPES_KEY, GNN_NODE_TYPES_KEY, GNN_NODE_LABELS_KEY


class GATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_dim, node_ft_out_dim, edge_ft_out_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g

        self.node_lv = NodeLevelGAT(g, node_dim, edge_dim, node_ft_out_dim, edge_ft_out_dim)
        z_node_lv_dim = node_ft_out_dim + edge_ft_out_dim  +  node_ft_out_dim
        # z_node_lv_dim = node_ft_out_dim + edge_ft_out_dim
        self.semantic_lv = SemLevelGAT(g, z_node_lv_dim, out_dim)


    def forward(self, node_features, edges_features, g):
        if g is not None:
            self.g = g

        z_node_lv, e_ft = self.node_lv(node_features, edges_features, self.g)
        z_final = self.semantic_lv(z_node_lv, self.g)

        return z_final, e_ft



class NodeLevelGAT(nn.Module):
    """
    Weight by neighbor nodes
    """
    def __init__(self, g, node_in_dim, edge_in_dim, node_ft_out_dim, edge_ft_out_dim):
        super(NodeLevelGAT, self).__init__()
        self.g = g

        # equation (1)
        self.fc_n = nn.Linear(node_in_dim, node_ft_out_dim, bias=False)
        self.fc_e = nn.Linear(edge_in_dim, edge_ft_out_dim, bias=False)
        print('`~~~edge_in_dim', edge_in_dim)
        # equation (2)
        self.attn_fc = nn.Linear(node_ft_out_dim + edge_ft_out_dim  +  node_ft_out_dim, 1, bias=False)
        # outdim
        # self.fc2 = nn.Linear(node_ft_out_dim + edge_ft_out_dim  +  node_ft_out_dim, node_ft_out_dim, bias=False)


        # self.attn_fc = nn.Linear(node_in_dim + edge_in_dim  +  node_in_dim, 1, bias=False)
        # self.fc2 = nn.Linear(node_in_dim + edge_in_dim, node_ft_out_dim, bias=False)

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
        src_node_n_edge_ft__fc = edges.data['zne']
        z = torch.cat([src_node_n_edge_ft__fc, edges.dst['z']], dim=1)
        return {'z_src': src_node_n_edge_ft__fc, 'z': z, 'e': edges.data['e'], 'edge_type': edges.data[GNN_EDGE_TYPES_KEY]}

    def reduce_func(self, nodes):

        # z = nodes.mailbox['z_src']
        z = nodes.mailbox['z']
        e = nodes.mailbox['e']
        e_type = nodes.mailbox['edge_type']

        e_type_shape = e_type.shape
        num_edge = e_type_shape[1]
        e_type_dim = e_type_shape[2]

        e_type_ = e_type.unsqueeze(0).view(e_type.shape[0], num_edge, e_type_dim, -1)


        # Equation (3)
        #   Normalize the attention scores using softmax (equation (3)).
        # print("nodes.mailbox['e']", nodes.mailbox['e'])
        alpha = F.softmax(e, dim=1)
        # print('alpha', alpha)
        # print('mailbox z', nodes.mailbox['z'].shape)
        # print('alpha shape', alpha.shape)

        weighted = alpha * z


        # Represent features by type so we can calculate semantic-level attention later
        weighted_rp = weighted.repeat(1, 1, e_type_dim).view(z.shape[0], z.shape[1], e_type_dim, -1)
        weighted_by_type = e_type_ * weighted_rp



        # Equation (4)
        #   Aggregate neighbor embeddings weighted by torche attention scores (equation(4)).
        h = torch.sum(weighted_by_type, dim=1)
        # h = torch.cat(h, nodes.data['z'], dim=1)

        # h = self.fc2(h)

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

        return self.g.ndata.pop('z'), self.g.edata.pop('e_ft')



class SemLevelGAT(nn.Module):
    """
    Weight by edge type
    """
    def __init__(self, g, z_node_lv_dim, out_dim):
        super(SemLevelGAT, self).__init__()
        self.g = g

        self.sem_attn = nn.Linear(z_node_lv_dim, 1, bias=False)
        self.fc2 = nn.Linear(z_node_lv_dim, out_dim, bias=False)

    def message_func(self, edges):
        return {'z': edges.src['z']}

    def update_node(self, nodes):
        h = nodes.data['z_final']
        zphi = torch.sum(h, dim=0)

        ''' Semantic-level attention '''
        w_phi = self.sem_attn(zphi)
        w_phi = F.leaky_relu(w_phi)
        beta = F.softmax(w_phi, dim=0)
        Z = torch.sum(beta * h, dim=1)

        Z = self.fc2(Z)

        return {'z_final': Z}


    def forward(self, h, g):
        if g is not None:
            self.g = g

        self.g.ndata['z_final'] = h

        self.g.apply_nodes(func=self.update_node)

        return self.g.ndata.pop('z_final')





class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_dim, node_ft_out_dim, edge_ft_out_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, node_dim, edge_dim, node_ft_out_dim, edge_ft_out_dim, out_dim))
        self.merge = merge

    def forward(self, node_features, edge_features, g):
        # print('self.heads~~~', self.heads)

        n_head_outs = []
        e_head_outs = []
        for attn_head in self.heads:
            n_head_out, e_head_out = attn_head(node_features, edge_features, g)
            n_head_outs.append(n_head_out)
            e_head_outs.append(e_head_out)

        if self.merge == 'cat':
            # concat on torche output feature dimension (dim=1)
            return torch.cat(n_head_outs, dim=1), torch.cat(e_head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(n_head_outs)), torch.mean(torch.stack(e_head_outs))
