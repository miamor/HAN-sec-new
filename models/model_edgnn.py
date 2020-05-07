"""
model_edgnn__0emb
"""
import copy
import importlib
import torch
import numpy as np
import scipy.sparse as sp

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import dgl
from dgl import DGLGraph
from utils.utils import compute_node_degrees
from utils.constants import *

import sys
import json
from utils.utils import preprocess_adj, care_APIs
from graphviz import Source
import networkx as nx
import matplotlib.pyplot as plt

# from models.layers.edgnn import edGNNLayer
# from models.layers.rgcn import RGCNLayer

ACTIVATIONS = {
    'relu': F.relu
}

class Model(nn.Module):

    type_weight = 4.0

    edge_features_use = 'all'
    node_features_use = 'all'

    def __init__(self, g, config_params, n_classes=None, n_rels=None, n_entities=None, is_cuda=False, seq_dim=None, batch_size=1, json_path=None, vocab_path=None, model_src_path=None, gdot_path=None):
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
        self.seq_dim = seq_dim # number of nodes in a sequence
        self.batch_size = batch_size

        print('======= g', g)

        # layer_type = config_params["layer_type"]
        self.model_src_path = model_src_path
        self.gdot_path = gdot_path

        self.build_model()


    def build_model(self):
        """
        Build NN
        """
        if self.model_src_path is not None:
            sys.path.insert(0, self.model_src_path)
            print('*** edgnn layer src path', self.model_src_path)
            from edgnn import edGNNLayer
        else:
            from models.layers.edgnn import edGNNLayer

        self.edgnn_layers = nn.ModuleList()
        layer_params = self.config_params['layer_params']        

        # basic tests
        assert (self.n_classes is not None)

        ############################
        # Build and append layers
        ############################
        print('\n*** Building model ***')

        ''' because self.g.ndata[GNN_NODE_LABELS_KEY] will be passed through Embedding layer '''
        if self.node_features_use == 'all':
            ''' use TYPES + LABELS as features '''
            self.node_dim = self.g.ndata[GNN_NODE_TYPES_KEY].shape[1] + self.g.ndata[GNN_NODE_LABELS_KEY].shape[1]
        elif self.node_features_use == 'type':
            ''' use TYPES as features '''
            self.node_dim = self.g.ndata[GNN_NODE_TYPES_KEY].shape[1]
        elif self.node_features_use == 'label':
            ''' use LABELS as features '''
            self.node_dim = self.g.ndata[GNN_NODE_LABELS_KEY].shape[1]

        if self.edge_features_use == 'all':
            ''' use TYPES + LABELS as features '''
            self.edge_dim = self.g.edata[GNN_EDGE_TYPES_KEY].shape[1] + self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]
            # self.edge_dim = self.g.edata[GNN_EDGE_TYPES_KEY].shape[1] + self.g.edata[GNN_EDGE_LABELS_KEY].shape[1] + self.g.edata[GNN_EDGE_BUFFER_SIZE_KEY].shape[1]
        elif self.edge_features_use == 'type':
            ''' use TYPES as features '''
            self.edge_dim = self.g.edata[GNN_EDGE_TYPES_KEY].shape[1]
        elif self.edge_features_use == 'label':
            ''' use LABELS as features '''
            self.edge_dim = self.g.edata[GNN_EDGE_LABELS_KEY].shape[1]

        print('self.node_dim, self.edge_dim', self.node_dim, self.edge_dim)

        """ edGNN layers """
        n_edGNN_layers = len(layer_params['n_units'])
        for i in range(n_edGNN_layers):
            if i == 0:  # take input from GAT layer
                print('* GNN:', self.node_dim, self.edge_dim,
                      layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]])

                edGNN = edGNNLayer(self.g, self.node_dim, self.edge_dim,
                                   layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]], is_cuda=self.is_cuda)
            else:
                print('* GNN:', layer_params['n_units'][i-1], self.edge_dim,
                      layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]])

                edGNN = edGNNLayer(self.g, layer_params['n_units'][i-1], self.edge_dim,
                                   layer_params['n_units'][i], ACTIVATIONS[layer_params['activation'][i]], is_cuda=self.is_cuda)

            # self.add_module('edGNN_{}'.format(i), edGNN)
            self.edgnn_layers.append(edGNN)
        
        """ Classification layer """
        print('* Building fc:', layer_params['n_units'][-1], self.n_classes)
        self.fc = nn.Linear(layer_params['n_units'][-1], self.n_classes)

        print('*** Model successfully built ***\n')


    def forward(self, g):
        if g is not None:
            g.set_n_initializer(dgl.init.zero_initializer)
            g.set_e_initializer(dgl.init.zero_initializer)
        self.g = g

        # print('~~~~~~~ self.g', self.g)

        ############################
        # 1. Build node features
        ############################
        # label features
        node_ft_lbl = self.g.ndata[GNN_NODE_LABELS_KEY].view(self.g.ndata[GNN_NODE_TYPES_KEY].shape[0], -1).type(torch.FloatTensor)
        # type features
        node_ft_type = self.g.ndata[GNN_NODE_TYPES_KEY].type(torch.FloatTensor)/self.type_weight

        if self.is_cuda:
            node_ft_type = node_ft_type.cuda()
            node_ft_lbl = node_ft_lbl.cuda()

        # combine features
        if self.node_features_use == 'all':
            node_features = torch.cat((node_ft_type, node_ft_lbl), dim=1)
        elif self.node_features_use == 'type':
            node_features = node_ft_type
        elif self.node_features_use == 'label':
            node_features = node_ft_lbl

        # print('\tnode_features', node_features)
        # print('\tnode_ft_lbl.shape', node_ft_lbl.shape)
        # print('\tnode_ft_type.shape', node_ft_type.shape)
        # print('\t node_features.shape', node_features.shape)

        ############################
        # 2. Build edge features
        ############################
        # label features
        edge_ft_lbl = self.g.edata[GNN_EDGE_LABELS_KEY].view(self.g.edata[GNN_EDGE_TYPES_KEY].shape[0], -1).type(torch.FloatTensor)
        # type features
        edge_ft_type = self.g.edata[GNN_EDGE_TYPES_KEY].type(torch.FloatTensor)/self.type_weight
        # buffer size features
        edge_ft_bufsize = self.g.edata[GNN_EDGE_BUFFER_SIZE_KEY].type(torch.FloatTensor)
        edge_ft_bufsize = edge_ft_bufsize.div(torch.max(edge_ft_bufsize))

        if self.is_cuda:
            edge_ft_type = edge_ft_type.cuda()
            edge_ft_lbl = edge_ft_lbl.cuda()
            edge_ft_bufsize = edge_ft_bufsize.cuda()

        # combine features
        if self.edge_features_use == 'all':
            # edge_features = torch.cat((edge_ft_lbl, edge_ft_type, edge_ft_bufsize), dim=1)
            edge_features = torch.cat((edge_ft_type, edge_ft_lbl), dim=1)
        elif self.edge_features_use == 'type':
            edge_features = edge_ft_type
        elif self.edge_features_use == 'label':
            edge_features = edge_ft_lbl

        # print('\tedge_features', edge_features)
        # print('\tedge_ft_lbl.shape', edge_ft_lbl.shape)
        # print('\tedge_ft_type.shape', edge_ft_type.shape)
        # print('\t edge_features.shape', edge_features.shape)

        #################################
        # 4. Iterate over each layer
        #################################
        for layer_idx, layer in enumerate(self.edgnn_layers):
            # print('~~ self.gdot_path', self.gdot_path)
            # if self.gdot_path is not None:
            #     layer.g_viz = nx.drawing.nx_pydot.read_dot(self.gdot_path)

            if layer_idx == 0:  # these are gat layers
                h = node_features
            # else:
                # h = self.g.ndata['h_'+str(layer_idx-1)]
            # h = h.type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
            h = layer(h, edge_features, self.g)
            # save only last layer output
            if layer_idx == len(self.edgnn_layers)-1:
                key = 'h_' + str(layer_idx)
                self.g.ndata[key] = h


        ####################################################
        # Visualize for inspection (only when inference)
        ####################################################
        # if self.gdot_path is not None:
        #     gdot_name = self.gdot_path.split('/')[-1]
        #     viz_dot_path = "/media/fitmta/Storage/MinhTu/HAN_sec_new/data/graphviz/data_report"
        #     G = layer.g_viz
        #     nx.drawing.nx_pydot.write_dot(G, '{}/{}.dot'.format(viz_dot_path, gdot_name))
        #     # pos = nx.spring_layout(G)
        #     # print('G', G)
        #     # print('pos', pos)
        #     # nx.draw(G, pos, with_labels=True)
        #     # nx.draw_networkx_edge_labels(G, pos)
        #     # plt.show(block=False)
        #     # plt.savefig('{}/{}.png'.format(viz_dot_path, gdot_name), format="PNG")

        #     # with open('{}/{}.dot'.format(viz_dot_path, gdot_name), 'r') as f:
        #     #     gsrc = f.read().replace('\n', '')
        #     #     gviz = Source(gsrc)
        #     #     gviz.render('{}/{}.png'.format(viz_dot_path, gdot_name), view=True)
        #     (graph,) = pydot.graph_from_dot_file('{}/{}.dot'.format(viz_dot_path, gdot_name))
        #     graph.write_png('{}/{}__weighted.png'.format(viz_dot_path, gdot_name))


        #############################################################
        # 5. It's graph classification, construct readout function
        #############################################################
        # sum with weights so that only features of last nodes is used
        last_layer_key = 'h_' + str(len(self.edgnn_layers)-1)

        # last_layer_out = self.g.ndata[last_layer_key]
        # a = self.last_attn_fc(last_layer_out)
        # # e = F.leaky_relu(a)
        # e = a
        # alpha = F.softmax(e, dim=1)
        # last_layer_out_weighted = alpha * last_layer_out

        # sum_node = dgl.sum_nodes(g, last_layer_key, 'w')
        sum_node = dgl.sum_nodes(g, last_layer_key)


        sum_node = sum_node.type(torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor)
        # print('\t h_last_node', h_last_node)
        # print('\t h_last_node.shape', h_last_node.shape)

        final_output = self.fc(sum_node)
        # print('final_output', final_output)
        final_output = F.softmax(final_output, dim=1)
        # print('final_output', final_output)
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

        if self.is_cuda is False:
            labels = labels.cpu()

        with torch.no_grad():
            logits = self(testing_graphs)
            loss = loss_fcn(logits, labels)
            _, indices = torch.max(logits, dim=1)
            corrects = torch.sum(indices == labels)
            
            return corrects.item() * 1.0 / len(labels), loss, logits
