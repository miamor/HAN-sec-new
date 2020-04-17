"""
Model Interface (gat_w) (with edge weighing and semantic)
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

from models.layers.gat_w import GATLayer, MultiHeadGATLayer


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
            self.edge_dim = self.g.edata[GNN_EDGE_TYPES_KEY].shape[1] + self.g.edata[GNN_NODE_LABELS_KEY].shape[1]
        elif self.edge_features_use == 'label':
            self.edge_dim = self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]

        # self.nodes_num = self.g.number_of_nodes()
        # print('self.node_dim', self.node_dim)
        # print('nodes_num', self.nodes_num)

        self.build_model()

    def build_model(self):
        """
        Build NN
        """
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
        # self.fc = nn.Linear(self.num_heads * self.gat_out_dim, self.n_classes)

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

        for layer_idx, gat_layer in enumerate(self.gat_layers):
            if layer_idx == 0:  # these are gat layers
                xn = node_features
                xe = edge_features

            xn, xe = gat_layer(xn, xe, self.g)
            # xn = gat_layer(xn, xe, self.g)
            if layer_idx < len(self.gat_layers) - 1:
                xn = F.leaky_relu(xn)
                xe = F.leaky_relu(xe)
            

        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        self.g.ndata['att_last'] = xn

        # print("self.g.ndata['att_last']", self.g.ndata['att_last'])
        
        # print('att_last shape', x.shape)

        #############################################################
        # 5. It's graph classification, construct readout function
        #############################################################
        # sum with weights so that only features of last nodes is used
        # last_layer_key = 'h_' + str(len(self.layers)-1)
        last_layer_key = 'att_last'
        sum_node = dgl.sum_nodes(self.g, last_layer_key)
        # print('\t sum_node', sum_node)
        # print('\t sum_node.shape', sum_node.shape)

        final_output = self.fc(sum_node)
        # print('final_output', final_output)
        final_output = F.softmax(final_output, dim=1)
        # final_output = F.sigmoid(final_output)
        # final_output = final_output.squeeze(1)
        print('sum_node', sum_node)
        print('final_output', final_output)
        # final_output = final_output.type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        # print('\t final_output.shape', final_output.shape)
        # print('\n')
        
        return final_output


    def eval_graph_classification(self, labels, testing_graphs):
        self.eval()
        loss_fcn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = self(testing_graphs)
            print('labels', labels)
            print('logits', logits)
            loss = loss_fcn(logits, labels)
            print('loss', loss)
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
