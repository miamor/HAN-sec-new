import time
import numpy as np
import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import json

from utils.early_stopping import EarlyStopping
from utils.io import load_checkpoint
from utils.utils import label_encode_onehot, indices_to_one_hot

from utils.constants import *
# from models.model import Model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import sys
from utils.utils import load_pickle, save_pickle, save_txt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# def collate(samples):
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels).cuda() if labels[0].is_cuda else torch.tensor(labels)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class App:
    """
    App inference
    """
    
    TRAIN_SIZE = 0.7

    def __init__(self, data, model_config, learning_config, pretrained_weight, early_stopping=True, patience=100, json_path=None, pickle_folder=None, vocab_path=None, mapping_path=None, odir=None, model_src_path=None, append_nid_eid=False, gdot_path=None):
        if model_src_path is not None:
            sys.path.insert(1, model_src_path)
            print('*** model_src_path', model_src_path)
            from model_edgnn_o import Model
        else:
            from models.model_edgnn_o import Model

        self.data = data
        self.model_config = model_config
        # max length of a sequence (max nodes among graphs)
        self.learning_config = learning_config
        self.pretrained_weight = pretrained_weight
        self.is_cuda = learning_config['cuda']

        # with open(vocab_path+'/../mapping.json', 'r') as f:
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)

        # print('[App][__init__] GNAMES', GNAMES)
        # print('[App][__init__] self.data', self.data)
        self.graphs_names = self.data[GNAMES]
        # print('[App][__init__] self.graphs_names', self.graphs_names)

        self.data_graph = self.data[GRAPH]

        # save nid and eid to nodes & edges
        if append_nid_eid is True:
            print('self.data_graph[0]', self.data_graph[0])
            if 'nid' not in self.data_graph[0].ndata:
                print('*** Not found nid. Appending...')
            # if True:
                for k,g in enumerate(self.data_graph):
                    g = self.write_nid_eid(g)
                    self.data_graph[k] = g
                # print('self.data_graph', self.data_graph)
            save_pickle(self.data_graph, os.path.join(pickle_folder, GRAPH))


        # data_nclasses = self.data[N_CLASSES]
        data_nclasses = 2
        if N_RELS in self.data:
            data_nrels = self.data[N_RELS]
        else:
            data_nrels = None
            
        if N_ENTITIES in self.data:
            data_nentities = self.data[N_ENTITIES]
        else:
            data_nentities = None

        self.model = Model(g=self.data_graph[0],
                           config_params=self.model_config,
                           n_classes=data_nclasses,
                           n_rels=data_nrels,
                           n_entities=data_nentities,
                           is_cuda=self.is_cuda,
                           batch_size=1,
                           model_src_path=model_src_path,
                           gdot_path=gdot_path)

        if self.is_cuda is True:
            # self.model.cuda()
            print('* Use cuda')
            self.model.to(torch.device('cuda'))


        # print('*** Model parameters ***')
        # pp = 0
        # print('self.model', self.model)
        
        # # self.e_src_attn_layer = self.model.edgnn_layers[1].e_src_attn_fc
        # # self.e_group_attn_layer = self.model.edgnn_layers[1].e_group_attn_fc
        # # # self.dst_attn_layer = self.model.edgnn_layers[1].dst_attn_fc
        # # print('self.e_group_attn_layer', self.e_group_attn_layer)
        # # print('self.e_src_attn_layer', self.e_src_attn_layer)
        # # # print('self.dst_attn_layer', self.dst_attn_layer)
        # # self.e_group_attn_layer.register_forward_hook(self.hook)
        # # self.e_src_attn_layer.register_forward_hook(self.hook2)
        # # self.dst_attn_layer.register_forward_hook(self.hook)

        # for p in list(self.model.parameters()):
        #     nn = 1
        #     for s in list(p.size()):
        #         # print('p', p)
        #         # print('\t s, nn, nn*s', s, nn, nn*s)
        #         nn = nn*s
        #     pp += nn
        # print('Total params', pp)


        if early_stopping:
            self.early_stopping = EarlyStopping(
                patience=patience, verbose=True)
            
        # Output folder to save train / test data
        if odir is None:
            odir = 'output/'+time.strftime("%Y-%m-%d_%H-%M-%S")
        self.odir = odir
    
    def load_data(self, data):
        self.data = data



    def write_nid_eid(self, g):
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        g.ndata['nid'] = torch.tensor([-1]*num_nodes)
        g.edata['eid'] = torch.tensor([-1]*num_edges)
        # print("self.g.ndata['nid']", g.ndata['nid'])
        # save nodeid and edgeid to each node and edge
        for nid in range(num_nodes):
            g.ndata['nid'][nid] = torch.tensor([nid]).type(torch.LongTensor)
        for eid in range(g.number_of_edges()):
            g.edata['eid'][eid] = torch.tensor([eid]).type(torch.LongTensor)
        return g


    def hook(self, module, inp, outp):
        # print('module', module)
        # print('len graph', len(self.data_graph))
        # print('enum', self.data_graph[0].edata['to_in_graph'].shape)
        # print('nnum', self.data_graph[0].ndata['nid'].shape)
        # print('outp', outp)
        print('inp', inp[0].shape)
        print('\toutp', outp.shape)
        print('\tsum outp', torch.sum(outp,dim=1))

    def hook2(self, module, inp, outp):
        # print('module', module)
        # print('len graph', len(self.data_graph))
        # print('enum', self.data_graph[0].edata['to_in_graph'].shape)
        # print('nnum', self.data_graph[0].ndata['nid'].shape)
        # print('outp', outp)
        print('(2) inp', inp[0].shape)
        print('\t(2) outp', outp.shape)
        alpha = F.softmax(outp, dim=1)
        o = alpha*inp[0]
        # print('\t(2) o', o.shape)
        print('\t(2) sum alpha', torch.sum(alpha))


    def load_model_state(self, model_path=''):
        try:
            print('[App][load_model_state] *** Load pre-trained model '+model_path+' ***')
            self.model = load_checkpoint(self.model, model_path, self.is_cuda)
        except ValueError as e:
            print('Error while loading the model.', e)

    def predict(self):
        print('-------- [App][predict] Predict -------- | len graphs', len(graphs))
        graphs = self.data[GRAPH]
        # print('*** len graphs', len(graphs))
        # print('[App][predict] self.data', self.data)

        # print('[App][predict] graphs', graphs)
        batches = dgl.batch(graphs)
        # print('[App][predict] batches', batches)
        
        # logits = self.model.classify(batches)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batches)

        logits = logits.cpu()
        # print('logits', logits)
        scores, indices = torch.max(logits, dim=1)
        return indices, scores

