import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY, GNN_EDGE_LABELS_KEY, GNN_EDGE_TYPES_KEY, GNN_NODE_TYPES_KEY, GNN_NODE_LABELS_KEY


class GATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_ft_out_dim, z_node_lv_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g

        edge_lbl_dim = self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]

        z_concat_edges_dim = node_dim + edge_ft_out_dim
        print('node_dim', node_dim)
        print('edge_ft_out_dim', edge_ft_out_dim)
        print('z_concat_edges_dim', z_concat_edges_dim)
        print('out_dim', out_dim)

        # self.weighted_agg_edge = WeightedAggEdge(g, node_dim, edge_lbl_dim, edge_ft_out_dim, z_concat_edges_dim)
        # self.node_lv = NodeLevelGAT(g, z_concat_edges_dim, z_node_lv_dim)
        # self.semantic_lv = SemLevelGAT(g, z_node_lv_dim, out_dim)

        # self.weighted_agg_edge = WeightedAggEdge(g, node_dim, edge_lbl_dim, edge_ft_out_dim, z_concat_edges_dim)
        # self.node_lv = NodeLevelGAT(g, z_concat_edges_dim, out_dim)

        self.node_lv = NodeLevelGAT(g, node_dim, out_dim)

        # self.node_lv = NodeLevelGAT(g, node_dim, z_node_lv_dim)
        # self.semantic_lv = SemLevelGAT(g, z_node_lv_dim, out_dim)


    def forward(self, node_features, g):
        if g is not None:
            self.g = g

        # weighted-sum edges features
        # z_concat_edges = self.weighted_agg_edge(node_features, g)
        # print('z_concat_edges', z_concat_edges.shape)

        # node-level
        # z_node_lv = self.node_lv(z_concat_edges, g)
        # print('z_node_lv', z_node_lv.shape)

        # semantic level
        # z_final = self.semantic_lv(z_node_lv, g)
        # print('z_final', z_final.shape)


        # z_concat_edges = self.weighted_agg_edge(node_features, g)
        # z_final = self.node_lv(z_concat_edges, g)

        z_final = self.node_lv(node_features, g)

        return z_final


class WeightedAggEdgeConcatNode(nn.Module):
    """
    Weight by edge features (edge_lbl_ft)
    """
    def __init__(self, g, node_dim, edge_lbl_dim, edge_ft_out_dim, z_concat_edges_dim):
        super(WeightedAggEdgeConcatNode, self).__init__()
        self.g = g
        
        input_concat_dim = node_dim + edge_ft_out_dim
        # z_concat_edges_dim actually = input_concat_dim, meaning we're keeping the size
        self.linear = nn.Linear(input_concat_dim, z_concat_edges_dim, bias=False)

    def gnn_msg(self, edges):
        # msg = edges.data[GNN_EDGE_LABELS_KEY]
        print("edges.src['node_feat']", edges.src['node_feat'])
        msg = torch.cat([edges.src['node_feat'], edges.data[GNN_EDGE_LABELS_KEY]], dim=1)
        return {GNN_MSG_KEY: msg}

    def gnn_reduce(self, nodes):
        # a = self.fc(nodes.mailbox[GNN_MSG_KEY])
        a = nodes.mailbox[GNN_MSG_KEY]

        accum = torch.sum((a), 1) / len(nodes)

        print('a', a)
        print('accum', accum)
        # print('nodes.mailbox[GNN_MSG_KEY]', nodes.mailbox[GNN_MSG_KEY])
        # print('nodes.mailbox[GNN_MSG_KEY]', nodes.mailbox[GNN_MSG_KEY].shape)
        # print('a', a.shape)
        print('accum', accum.shape)
        print('\n')

        return {GNN_AGG_MSG_KEY: accum}

    def node_update(self, nodes):
        # print('nodes', nodes)
        # print("nodes.data['node_feat'].shape", nodes.data['node_feat'].shape)
        # print("nodes.data[GNN_AGG_MSG_KEY].shape", nodes.data[GNN_AGG_MSG_KEY].shape)
        h = nodes.data[GNN_AGG_MSG_KEY] # output_dim = node_dim + edge_ft_out_dim
        # print('h.shape', h.shape)
        # print('\n')
        h = self.linear(h)

        h = F.elu(h)

        return {'z_concat_edges': h} # dim = node_dim + edge_ft_out_dim


    def forward(self, h, g):
        if g is not None:
            self.g = g

        e = self.g.edata[GNN_EDGE_LABELS_KEY]
        # print('e (WeightedAggEdge)', e.shape)

        self.g.ndata['node_feat'] = h

        # weighted-aggregation of edges
        # h(i) = concat(h(i), weighted_sum_edges_ft)
        self.g.update_all(self.gnn_msg,
                          self.gnn_reduce,
                          self.node_update)

        return self.g.ndata.pop('z_concat_edges')




class WeightedAggEdge(nn.Module):
    """
    Weight by edge features (edge_lbl_ft)
    """
    def __init__(self, g, node_dim, edge_lbl_dim, edge_ft_out_dim, z_concat_edges_dim):
        super(WeightedAggEdge, self).__init__()
        self.g = g
        
        self.fc = nn.Linear(edge_lbl_dim, edge_ft_out_dim, bias=False)
        
        # print('edge_ft_out_dim', edge_ft_out_dim)
        # self.norm = nn.LayerNorm(edge_ft_out_dim)

        # input_concat_dim = node_dim + edge_ft_out_dim
        # z_concat_edges_dim = input_concat_dim # just keep the dim
        # self.linear = nn.Linear(input_concat_dim, z_concat_edges_dim, bias=False)

    def gnn_msg(self, edges):
        msg = edges.data[GNN_EDGE_LABELS_KEY]

        return {'e_lbl': msg}

    def gnn_reduce(self, nodes):
        # a = self.fc(nodes.mailbox['e_lbl'])
        a = nodes.mailbox['e_lbl']

        # print('a', a.shape)

        accum = torch.sum((a), 1) / len(nodes)
        # accum = torch.sum((a), 1)
        # accum = self.norm(accum)

        # print('a', a)
        # print('accum', accum)
        # print('nodes.mailbox['e_lbl']', nodes.mailbox['e_lbl'])
        # print('nodes.mailbox['e_lbl']', nodes.mailbox['e_lbl'].shape)
        # print('a', a.shape)
        # print('accum', accum.shape)
        # print('\n')

        # print('nodes', nodes)
        # print("nodes.data['node_feat']", nodes.data['node_feat'])
        # print("nodes.data[GNN_AGG_MSG_KEY]", nodes.data[GNN_AGG_MSG_KEY])
        h = torch.cat([nodes.data['node_feat'],
                       accum],
                      dim=1) # output_dim = node_dim + edge_ft_out_dim
        # print('h', h)

        # h = self.linear(h)
        # print('h lineared', h)

        h = F.elu(h)
        # print('h relu', h)
        # print('\n')

        return {'z_concat_edges': h} # dim = node_dim + edge_ft_out_dim


    def forward(self, h, g):
        if g is not None:
            self.g = g

        e = self.g.edata[GNN_EDGE_LABELS_KEY]
        # print('e (WeightedAggEdge)', e.shape)

        self.g.ndata['node_feat'] = h

        # weighted-aggregation of edges
        # h(i) = concat(h(i), weighted_sum_edges_ft)
        self.g.update_all(self.gnn_msg, self.gnn_reduce)

        return self.g.ndata.pop('z_concat_edges')



class NodeLevelGAT(nn.Module):
    """
    Weight by neighbor nodes
    """
    def __init__(self, g, node_ft_dim, z_node_lv_dim):
        super(NodeLevelGAT, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(node_ft_dim, z_node_lv_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * z_node_lv_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)        
        z2 = torch.cat([edges.src['z_concat_edges'], edges.dst['z_concat_edges']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z_concat_edges'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # Equation (3)
        #   Normalize the attention scores using softmax (equation (3)).
        # print("nodes.mailbox['e']", nodes.mailbox['e'])
        alpha = F.softmax(nodes.mailbox['e'], dim=1)

        # Equation (4)
        #   Aggregate neighbor embeddings weighted by the attention scores (equation(4)).
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'z_node_lv': h}


    def forward(self, h, g):
        if g is not None:
            self.g = g

        # node-level
        # Equation (1)
        z = self.fc(h)
        self.g.ndata['z_concat_edges'] = z

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

        return self.g.ndata.pop('z_node_lv')



class SemLevelGAT(nn.Module):
    """
    Weight by edge type
    """
    def __init__(self, g, z_node_lv_dim, out_dim):
        super(SemLevelGAT, self).__init__()
        self.g = g
        # equation (1)
        edge_type_dim = self.g.edata[GNN_EDGE_TYPES_KEY].shape[1]
        self.attn_fc = nn.Linear(edge_type_dim, 1, bias=False)
        self.linear = nn.Linear(z_node_lv_dim, out_dim, bias=False)

    def edge_weight(self, edges):
        # edge UDF for equation (2)
        a = self.attn_fc(self.g.edata[GNN_EDGE_TYPES_KEY])
        return {'e_type_tanh': torch.tanh(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z_node_lv': edges.src['z_node_lv'], 'e_type_tanh': edges.data['e_type_tanh']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # print("len nodes.mailbox['e_type_tanh']", len(nodes.mailbox['e_type_tanh']))
        # print("len nodes", len(nodes))
        sum_tanh = torch.sum(nodes.mailbox['e_type_tanh'], dim=1)
        e_type_w = sum_tanh/len(nodes)

        # print("nodes.mailbox['e_type_tanh']", nodes.mailbox['e_type_tanh'])
        # print('sum tanh', sum_tanh)
        # print('e_type_w', e_type_w)
        # Equation (3)
        #   Normalize the attention scores using softmax (equation (3)).
        beta = F.softmax(e_type_w, dim=1)
        rows = beta.shape[0]
        # beta = torch.transpose(beta, 0, 1)
        beta = beta.unsqueeze(1)

        # print('beta', beta)
        # print("nodes.mailbox['z_node_lv']", nodes.mailbox['z_node_lv'])

        # print("nodes.mailbox['z_node_lv']", nodes.mailbox['z_node_lv'].shape)
        # print('beta', beta.shape)

        # Equation (4)
        #   Aggregate neighbor embeddings weighted by the attention scores (equation(4)).
        h = torch.sum(beta * nodes.mailbox['z_node_lv'], dim=1)
        # print('~!~~~h.shape', h.shape)
        z_final = self.linear(h)

        return {'z_final': z_final}


    def forward(self, h, g): # h: z_noed_lv
        if g is not None:
            self.g = g

        self.g.ndata['z_node_lv'] = h

        self.g.apply_edges(self.edge_weight)

        self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('z_final')





class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, node_dim, edge_ft_out_dim, z_node_lv_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, node_dim, edge_ft_out_dim, z_node_lv_dim, out_dim))
        self.merge = merge

    def forward(self, node_features, g):
        # print('self.heads~~~', self.heads)

        # head_outs = []
        # for attn_head in self.heads:
        #     head_out = attn_head(node_features, g)
        #     head_outs.append(head_out)
        #     print('head_out~~~', head_out)
        
        head_outs = [attn_head(node_features, g) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))
