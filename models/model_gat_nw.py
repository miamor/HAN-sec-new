"""
Model Interface (gat_nw) (no edge_weighing, with semantic)
"""
import copy
import importlib
import torch
import numpy as np
import scipy.sparse as sp
from utils.utils import preprocess_adj

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import dgl
from dgl import DGLGraph
from utils.utils import compute_node_degrees
from utils.constants import *

from models.layers.gat_nw import GATLayer, MultiHeadGATLayer


class Model(nn.Module):

    # node_features_use = 'label'
    node_features_use = 'all'
    edge_features_use = 'label'

    def __init__(self, g, config_params, n_classes=None, n_rels=None, n_entities=None, is_cuda=False, batch_size=1, json_path=None, vocab_path=None):
        """
        Instantiate a graph neural network.

        Args:
            g (DGLGraph): a preprocessed DGLGraph
            config_json (str): path to a configuration JSON file. It must contain the following fields: 
                               "layer_type", and "layer_params". 
                               The "layer_params" should be a (nested) dictionary containing at least the fields 
                               "n_units" and "activation". "layer_params" should contain other fields that corresponds
                               to keyword arguments of the concrete layers (refer to the layers implementation).
                               The name of these additional fields should be the same as the keyword args names.
                               The parameters in "layer_params" should either be lists with the same number of elements,
                               or single values. If single values are specified, then a "n_hidden_layers" (integer) 
                               field is expected.
                               The fields "n_input" and "n_classes" are required if not specified 
        """
        super(Model, self).__init__()

        self.is_cuda = is_cuda
        self.config_params = config_params
        self.n_rels = n_rels
        self.n_classes = n_classes
        self.n_entities = n_entities
        self.g = g
        # merge all graphs

        # self.seq_dim = seq_dim # number of nodes in a sequence
        self.batch_size = batch_size

        # print('self.g', self.g)
        # print('self.g.ndata', self.g.ndata)
        if self.node_features_use == 'all':
            self.node_dim = self.g.ndata[GNN_NODE_TYPES_KEY].shape[1] + self.g.ndata[GNN_NODE_LABELS_KEY].shape[1]
        elif self.node_features_use == 'label':
            self.node_dim = self.g.ndata[GNN_NODE_LABELS_KEY].shape[1]

        if self.edge_features_use == 'all':
            self.edge_dim = self.g.edata[GNN_EDGE_TYPES_KEY].shape[1] + self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]
        elif self.edge_features_use == 'label':
            self.edge_dim = self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]

        # self.edge_lbl_dim = self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]

        self.num_heads = 1
        self.gat_out_dim = 4

        # self.nodes_num = self.g.number_of_nodes()
        # print('self.node_dim', self.node_dim)
        # print('nodes_num', self.nodes_num)

        self.build_model()

    def build_model(self):
        """
        Build NN
        """
        # self.layer1 = MultiHeadGATLayer(self.g, self.node_dim, edge_ft_out_dim=self.edge_lbl_dim, z_node_lv_dim=3, out_dim=self.hidden_dim, num_heads=self.num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        # self.layer2 = MultiHeadGATLayer(self.g, self.num_heads * self.hidden_dim, edge_ft_out_dim=self.edge_lbl_dim, z_node_lv_dim=3, out_dim=self.gat_out_dim, num_heads=self.num_heads)
        
        # self.layer1 = MultiHeadGATLayer(self.g, self.node_dim, self.edge_dim, out_dim=self.gat_out_dim, num_heads=self.num_heads)
        # self.layer2 = MultiHeadGATLayer(self.g, self.num_heads * self.gat_out_dim, self.edge_dim, out_dim=self.gat_out_dim, num_heads=self.num_heads)
        # self.layer3 = MultiHeadGATLayer(self.g, self.num_heads * self.gat_out_dim, self.edge_dim, out_dim=self.gat_out_dim, num_heads=self.num_heads)

        print('\n*** Building model ***')
        self.gat_layers = nn.ModuleList()
        layer_params = self.config_params['layer_params']

        n_gat_layers = len(layer_params['n_heads'])

        for i in range(n_gat_layers):
            if i == 0:  # take input from GAT layer
                node_in_dim = self.node_dim
                edge_in_dim = self.edge_dim
            else:
                node_in_dim = layer_params['hidden_dim'][i-1] * layer_params['n_heads'][i-1]
                edge_in_dim = layer_params['e_hidden_dim'][i-1] * layer_params['n_heads'][i-1]
                # edge_in_dim = layer_params['e_hidden_dim'][i-1]
                # edge_in_dim = self.edge_lbl_dim

            print('* GAT (in_dim, out_dim, num_heads):', node_in_dim, layer_params['n_hidden_dim'][i], layer_params['e_hidden_dim'][i], layer_params['hidden_dim'][i], layer_params['n_heads'][i])

            gat = MultiHeadGATLayer(self.g, node_dim=node_in_dim, edge_dim=edge_in_dim, node_ft_out_dim=layer_params['n_hidden_dim'][i], edge_ft_out_dim=layer_params['e_hidden_dim'][i], out_dim=layer_params['hidden_dim'][i], num_heads=layer_params['n_heads'][i])

            self.gat_layers.append(gat)



        """ Classification layer """
        # print('* Building fc layer with args:', layer_params['n_units'][-1], self.n_classes)
        self.fc = nn.Linear(layer_params['n_heads'][-1]*layer_params['hidden_dim'][-1], self.n_classes)

        print('*** Model successfully built ***\n')


    def forward(self, g):
        # print(g)

        if g is not None:
            g.set_n_initializer(dgl.init.zero_initializer)
            g.set_e_initializer(dgl.init.zero_initializer)
            self.g = g

        ############################
        # 1. Build node features
        ############################
        # node_features = self.g.ndata[GNN_NODE_LABELS_KEY]
        self.g.ndata[GNN_NODE_TYPES_KEY] = self.g.ndata[GNN_NODE_TYPES_KEY].type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        self.g.ndata[GNN_NODE_LABELS_KEY] = self.g.ndata[GNN_NODE_LABELS_KEY].type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor).view(self.g.ndata[GNN_NODE_TYPES_KEY].shape[0], -1)

        if self.node_features_use == 'all':
            node_features = torch.cat((self.g.ndata[GNN_NODE_TYPES_KEY], self.g.ndata[GNN_NODE_LABELS_KEY]), dim=1)
        elif self.node_features_use == 'label':
            node_features = self.g.ndata[GNN_NODE_LABELS_KEY]

        # print('\tnode_features', node_features)
        # node_features = node_features.view(node_features.size()[0], -1)
        # self.node_dim = node_features.size()[1]
        # print('\tnode_features', node_features)
        # print('\tnode_features.shape', node_features.shape)

        ############################
        # 2. Build edge features
        ############################
        self.g.edata[GNN_EDGE_TYPES_KEY] = self.g.edata[GNN_EDGE_TYPES_KEY].type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        self.g.edata[GNN_EDGE_LABELS_KEY] = self.g.edata[GNN_EDGE_LABELS_KEY].type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        
        if self.edge_features_use == 'all':
            edge_features = torch.cat((self.g.edata[GNN_EDGE_TYPES_KEY], self.g.edata[GNN_EDGE_LABELS_KEY]), dim=1)
        elif self.edge_features_use == 'label':
            edge_features = self.g.edata[GNN_EDGE_LABELS_KEY]

        # edge_features = self.g.edata[GNN_EDGE_LABELS_KEY]
        
        # edge_features = edge_features.view(edge_features.size()[0], -1)
        # self.edge_dim = edge_features.size()[1]
        # print('\tedge_features', edge_features)
        # print('\tedge_features.shape', edge_features.shape)

        ############################
        # 3. Calculate adj matrix
        ############################
        # nodes_idx = self.g.nodes()
        # n_nodes = self.g.number_of_nodes()

        # edges_src, edges_dst = self.g.edges()
        # edges_src = list(edges_src.data.numpy())
        # edges_dst = list(edges_dst.data.numpy())

        # adj = np.zeros((n_nodes, n_nodes))
        # for src, dst in zip(edges_src, edges_dst):
        #     adj[src][dst] = 1

        # adj = sp.coo_matrix(adj, dtype=np.float32)
        # # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = preprocess_adj(adj, symmetric=True)  # sparse
        # adj = adj.todense()
        # # print('adj', adj)
        # adj = torch.tensor(adj).type(torch.cuda.FloatTensor if self.cuda else torch.FloatTensor)

        #################################
        # 4. Iterate over each layer
        #################################
        # for layer_idx, layer in enumerate(self.layers):
        #     if layer_idx == 0:  # these are gat layers
        #         h = node_features
        #     # else:
        #         # h = self.g.ndata['h_'+str(layer_idx-1)]
        #     h = h.type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        #     h = layer(h, edge_features, self.g)
        #     key = 'h_' + str(layer_idx)
        #     self.g.ndata[key] = h

        # x = F.dropout(x, self.dropout_gat, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout_gat, training=self.training)

        # print('g.ndata[GNN_NODE_TYPES_KEY].shape', g.ndata[GNN_NODE_TYPES_KEY].shape)

        # x = self.layer1(node_features, edge_features, g)
        # x = F.leaky_relu(x)
        # x = self.layer2(x, edge_features, g)
        # x = F.elu(x)
        # x = self.layer3(x, edge_features, g)

        for layer_idx, gat_layer in enumerate(self.gat_layers):
            if layer_idx == 0:  # these are gat layers
                xn = node_features
                xe = edge_features

            xn, xe = gat_layer(xn, xe, g)
            if layer_idx < len(self.gat_layers) - 1:
                xn = F.leaky_relu(xn)
                xe = F.leaky_relu(xe)
            

        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        self.g.ndata['att_last'] = xn
        
        # print('att_last shape', x.shape)

        #############################################################
        # 5. It's graph classification, construct readout function
        #############################################################
        # sum with weights so that only features of last nodes is used
        # last_layer_key = 'h_' + str(len(self.layers)-1)
        last_layer_key = 'att_last'
        sum_node = dgl.sum_nodes(g, last_layer_key)
        # print('\t sum_node', sum_node)
        # print('\t sum_node.shape', sum_node.shape)

        final_output = self.fc(sum_node)
        # print('final_output', final_output)
        final_output = F.softmax(final_output, dim=1)
        # print('final_output softmax', final_output)
        # final_output = final_output.type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        # print('\t final_output.shape', final_output.shape)
        # print('\n')
        
        return final_output


    def eval_node_classification(self, labels, mask):
        self.eval()
        loss_fcn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = self(None)
            logits = logits[mask]
            labels = labels[mask]
            loss = loss_fcn(logits, labels)
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels), loss

    def eval_graph_classification(self, labels, testing_graphs):
        self.eval()
        loss_fcn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = self(testing_graphs)
            loss = loss_fcn(logits, labels)
            _, indices = torch.max(logits, dim=1)
            corrects = torch.sum(indices == labels)
            # print('labels', labels)
            # print('corrects', corrects)
            return corrects.item() * 1.0 / len(labels), loss, logits


    def classify(self, testing_graphs):
        self.eval()
        with torch.no_grad():
            logits = self(testing_graphs)
            return logits
