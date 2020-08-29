"""
prep_data__doc2vec_edge_node
"""
import sys
sys.path.insert(1, '../..')
from utils.utils import indices_to_one_hot, label_encode_onehot, sample_mask, preprocess_adj, preprocess_features, save_pickle, load_pickle, save_txt, care_APIs

import os
import json
import numpy as np
import scipy.sparse as sp
from dgl import DGLGraph
import math
import random
from utils.constants import *
import shutil

import torch
import torch.nn as nn

from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt
from utils.word_embedding import TFIDF, Doc2Vec_

class PrepareData(object):
    # cuckoo_analysis_dir = '/home/mtaav/.cuckoo/storage/analyses'
    cuckoo_analysis_dir = '/media/tunguyen/TuTu_Passport/MTAAV_data/HAN-sec-new/cuckoo-reports'
    DATA_OUT_PATH = ''  # data root dir
    # folder contains pickle file (save data in dortmund format)
    pickle_folder = ''

    # consider only apis that fall into these categories
    # allow_cat = ['network', 'file', 'registry', 'process']
    allow_cat = ['file', 'process', 'registry']

    use_interesting_apis = None
    mapping_labels = {'benign': 0, 'malware': 1}

    nb_attributes = 100  # size of feature space

    # path_type_code = {
    #     'proc_reg': 0,
    #     'proc_file': 1,
    #     'proc_network': 2,
    #     'proc_process': 3
    # }
    # node_type_code = {
    #     'proc': 0, # process_handle
    #     'file': 1, # file_handle
    #     'reg': 2, # registry key_handle

    #     # 'network': 2,
    #     'process_api': 3,
    #     'file_api': 4,
    #     'reg_api': 5,
    # }
    path_type_code = {
        'proc_reg': 0,
        'proc_file': 1,
        'proc_process': 2
    }
    node_type_code = {
        'proc': 0, # process_handle
        'file': 1, # file_handle
        'reg': 2, # registry key_handle

        # 'network': 2,
        'process_api': 3,
        'file_api': 4,
        'reg_api': 5,
    }
    node_color = ['red', 'orange', 'blue', 'pink', 'yellow', 'cyan']

    interesting_apis = care_APIs() + list(node_type_code.keys()) + ['other']
    # create_apis = ['Open', 'Create', 'Set', 'Write']


    from_folder = False

    def __init__(self, config):
        self.reset()

        # self.vocab_path = vocab_path.split('.txt')[0]
        self.vocab_path = config['vocab_path']
        # self.vocab_path = vocab_path
        self.vocab_path_node = self.vocab_path+'/node.txt'
        self.vocab_path_edge = self.vocab_path+'/edge.txt'
        
        self.save_json = True

        self.use_interesting_apis = config['use_interesting_apis']
        self.prepend_vocab = config['prepend_vocab']
        self.mapping_path = config['mapping_path']

        print('self.prepend_vocab', self.prepend_vocab)
        print('self.use_interesting_apis', self.use_interesting_apis)
        print('self.mapping_path', self.mapping_path)

        if config['input_report_folder'] is not None:
            self.reports_parent_dir_path = config['input_report_folder']

        if config['input_data_folder'] is not None:
            self.final_json_path = config['input_data_folder']
            data_dir = os.path.dirname(self.final_json_path)
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)
                
            json_data_dir = self.final_json_path
            if not os.path.isdir(json_data_dir):
                os.makedirs(json_data_dir)
            for key in self.json_data_paths:
                self.json_data_paths[key] = json_data_dir+'/'+key+'.json'
                

        if config['input_pickle_folder'] is not None:
            self.pickle_folder = config['input_pickle_folder']
            if not os.path.isdir(self.pickle_folder):
                os.makedirs(self.pickle_folder)


        if self.mapping_path is not None:
            with open(self.mapping_path) as json_file:
                self.mapping = json.load(json_file)
                for cls_name in self.mapping:
                    self.edge_args_embedding_data_csv__cls[cls_name] = []
                    self.node_name_embedding_data_csv__cls[cls_name] = []

            with open(self.pickle_folder+'/mapping.json', 'w') as f:
                json.dump(self.mapping, f)
        

        self.train_embedder = config['train_embedder'] if 'train_embedder' in config else False

        self.edge_args_emb_trained_path = config['train_embedding_path']+'/edge_csv'
        if not os.path.exists(self.edge_args_emb_trained_path):
            os.makedirs(self.edge_args_emb_trained_path)
        self.edge_args_emb_corpus_path = self.pickle_folder+'/edge_csv'
        if not os.path.exists(self.edge_args_emb_corpus_path):
            os.makedirs(self.edge_args_emb_corpus_path)

        self.node_name_emb_trained_path = config['train_embedding_path']+'/node_csv'
        if not os.path.exists(self.node_name_emb_trained_path):
            os.makedirs(self.node_name_emb_trained_path)
        self.node_name_emb_corpus_path = self.pickle_folder+'/node_csv'
        if not os.path.exists(self.node_name_emb_corpus_path):
            os.makedirs(self.node_name_emb_corpus_path)


        max_ft = config['max_ft']
        top_k = config['top_k']
        vector_size = config['vector_size']
        dm = config['dm']
        self.prepare_word_embedding = config['prepare_word_embedding']


        if 'train_list_file' in config and os.path.exists(config['train_list_file']):
            with open(config['train_list_file'], 'r') as f:
                lines = f.read().split('\n')
                self.train_list_name = [line.strip() for line in lines]
            with open(config['test_list_file'], 'r') as f:
                lines = f.read().split('\n')
                self.test_list_name = [line.strip() for line in lines]

        if config['edge_embedder'] == 'doc2vec':
            self.edge_embedder = Doc2Vec_(self.edge_args_emb_trained_path, self.edge_args_emb_corpus_path, self.mapping, vector_size, dm)
        elif config['edge_embedder'] == 'tfidf':
            self.edge_embedder = TFIDF(self.edge_args_emb_trained_path, self.edge_args_emb_corpus_path, self.mapping, max_ft, top_k)

        if config['node_embedder'] == 'doc2vec':
            self.node_embedder = Doc2Vec_(self.node_name_emb_trained_path, self.node_name_emb_corpus_path, self.mapping, vector_size, dm)
        elif config['node_embedder'] == 'tfidf':
            self.node_embedder = TFIDF(self.node_name_emb_trained_path, self.node_name_emb_corpus_path, self.mapping, max_ft, top_k)


        print('self.edge_args_emb_trained_path', self.edge_args_emb_trained_path)
        print('self.edge_args_emb_corpus_path', self.edge_args_emb_corpus_path)

        # self.edge_embedder = Doc2Vec_(self.edge_args_emb_trained_path, self.edge_args_emb_corpus_path, self.mapping, vector_size, dm)
        # self.node_embedder = Doc2Vec_(self.node_name_emb_trained_path, self.node_name_emb_corpus_path, self.mapping, vector_size, dm)

        return

    def reset(self):
        self.reports_parent_dir_path = ''  # path to report.json generated by cuckoo
        self.behavior = {}  # behavior extracted from report.json in json format

        self.edges = None
        self.nodes_labels = []
        self.edges_labels = []
        self.graphs_labels = []

        self.graphs_dict = dict()
        self.graphs_name_to_label = dict()
        self.graphs = []
        self.graphs_names = []
        self.graphs_viz = dict()

        self.final_json_data = {
            'nodes': [],
            'paths': []
        }
        self.final_json_path = ''  # path to data.json generated by process_txt() function
        
        self.json_data_paths = {
            'nodes': '',
            'proc_process': '',
            'proc_file': '',
            'proc_reg': '',
            'proc_network': ''
        }
        self.json_data = {
            'nodes': {},
            'proc_process': {},
            'proc_file': {},
            'proc_reg': {},
            'proc_network': {}
        }
        
        self.word_dict_node = []
        self.word_dict_edge = []

        self.current_edge_id = -1
        self.current_node_id = -1
        self.current_node_id_of_current_graph = -1
        self.api_nodes_existed_id = {}
        self.api_nodes_existed = {}


        self.handle_to_pid = {}
        self.handle_to_node = {}
        self.pid_to_node = {}

        self.data_dortmund_format = {}
        self.max_n_nodes = 0

        self.word_to_ix_node = {}
        self.word_to_ix_edge = {}

        self.edge_args_embedding_data_csv = []
        self.edge_args_embedding_data_csv__cls = {}
        self.node_name_embedding_data_csv = []
        self.node_name_embedding_data_csv__cls = {}
        self.edge_args_by_graph = {}
        self.node_names_by_graph = {}
        self.flags_keys = []
        self.mapping = {}


    def load_data(self, from_folder=False, from_json=False, from_pickle=False):
        """
        Return self.data_dortmund_format
        """
        self.has_label = True
        self.from_folder = from_folder
        if from_pickle is True:
            print('\n[load_data] Train & load node/edge embedding')
            # self.train_and_load_embedding()

            print('[load_data] Load data from pickle folder')
            self.load_from_pickle()
        else:
            ''' Copy prep_data file to this data folder '''
            if self.final_json_path is not None:
                shutil.copy('./utils/prep_data.py', self.final_json_path+'/../prep_data.py')

            if from_folder is True:
                if from_json is False:
                    self.save_json = True
                self.prepare_word_embedding = True
                
                print('\n[load_data] Process data from reports folder to data.json and encode nodes & edges')
                self.encode_reports_from_dir(self.save_json)

                print('\n[load_data] Gen corpus')
                self.gen_vocab_and_corpus()

            elif from_json is True:
                print('\n[load_data] Load data from json file')
                self.load_data_json()


            print('\n[load_data] Train & load node/edge embedding')
            self.train_and_load_embedding()

            if from_json is True:
                print('\n[load_data] Encode nodes & edges')
                self.encode_data()
            
            print('\n[load_data] Create graphs and save to pickle files from encoded data')
            self.create_graphs()

            # Use when handling with large dataset like pack1 (run in partition)
            # if from_json is True:
            #     print('\nEncode nodes & edges and create seperate graphs')
            #     for report_dir_name in os.listdir(self.reports_parent_dir_path): # category name (benign/malware)
            #         report_dir_path = os.path.join(self.reports_parent_dir_path, report_dir_name)
            #         for report_file_name in os.listdir(report_dir_path):
            #             g_name = '{}__{}'.format(report_dir_name, report_file_name)
            #             self.create_graph_from_report(g_name, report_dir_name)
            
            # print('\nMerge graphs and save to pickle files from encoded data')
            # self.merge_graphs()

        return self.data_dortmund_format


    def load_data_files(self, task_ids, report_file_name='report.json', write_to_json=True):
        """
        Preprocess data from cuckoo report
        report_dir_name: task_id
        Return self.data_dortmund_format
        """
        print('\n[load_data_files] Process data from one report file to json format and encode nodes & edges')
        self.has_label = False
        if task_ids is None:
            report_dir_name = 'malware'
            report_file_name = '0bee54efca5e41b8bcafcba4f70c9498d48e088553f7b442ea54670db9f874b2.json'

            report_path = '/media/fitmta/Storage/MinhTu/Dataset/one/'+report_dir_name+'/'+report_file_name

            print('[load_data_files] report_file_name', report_dir_name, report_file_name)
            print('[load_data_files] report_path', report_path)
            behavior = self.read_report(report_path)

            if behavior is not None:
                self.encode_report(behavior, report_file_name, 't'+report_dir_name)
            else:
                print('behavior none. Skip '+report_dir_name+'/'+report_file_name)
        else:
            for task_id in task_ids:
                report_dir_name = str(task_id)
                report_path = '{}/{}/reports/{}'.format(self.cuckoo_analysis_dir, report_dir_name, report_file_name)

                print('[load_data_files] report_file_name', report_dir_name, report_file_name)
                print('[load_data_files] report_path', report_path)
                behavior = self.read_report(report_path)

                if behavior is not None:
                    self.encode_report(behavior, report_file_name, 't'+report_dir_name)
                else:
                    print('[load_data_files] behavior none. Skip '+report_dir_name+'/'+report_file_name)

        self.gen_edge_args_embedding_data()
        self.gen_node_name_embedding_data()

        ''' Save processed json '''
        if write_to_json is True:
            print('[load_data_files] Process done. Saving to json file...')
            # Save json data to file
            # it's gonna be too large, use separate file to store information as belows
            n_empty_obj = 0
            for key in self.json_data_paths:
                with open(self.json_data_paths[key], 'w') as outfile:
                    json.dump(self.json_data[key], outfile)
                    # lenn += len(self.json_data[key])
                    if bool(self.json_data[key]) is False:
                        n_empty_obj += 1
            if n_empty_obj == len(self.json_data):
                print('G[load_data_files] raph cant be created')
                return None
        else:
            print('[load_data_files] Writing to json file option set to False. Skip saving.')

        print('\n[load_data_files] Train & load node/edge embedding')
        self.train_embedder = False
        self.from_folder = False
        self.train_and_load_embedding()

        print('\n[load_data_files] Encode nodes & edges')
        self.encode_data()
        

        print('\n[load_data_files] Create graphs and save to pickle files from encoded data')
        self.create_graphs()

        return self.data_dortmund_format



    def encode_reports_from_dir(self, write_to_json=True):
        for report_dir_name in os.listdir(self.reports_parent_dir_path):
            report_dir_path = os.path.join(
                self.reports_parent_dir_path, report_dir_name)
            n = 0
            for report_file_name in os.listdir(report_dir_path):
                n += 1
                print('[encode_reports_from_dir]', n, 'report_file_name', report_dir_name+'/'+report_file_name)
                behavior = self.read_report(os.path.join(
                    report_dir_path, report_file_name))

                if behavior is not None:
                    self.encode_report(
                        behavior, report_file_name, report_dir_name)
                else:
                    print('[encode_reports_from_dir] behavior none. Skip ' +
                          report_dir_name+'/'+report_file_name)
        
        self.gen_edge_args_embedding_data()
        self.gen_node_name_embedding_data()

        ''' Save processed json '''
        if write_to_json is True:
            print('[encode_reports_from_dir] Process done. Saving to json file...')
            # Save json data to file
            # it's gonna be too large, use separate file to store information as belows
            for key in self.json_data_paths:
                with open(self.json_data_paths[key], 'w') as outfile:
                    json.dump(self.json_data[key], outfile)
        else:
            print('[encode_reports_from_dir] Writing to json file option set to False. Skip saving.')



    def train_and_load_embedding(self):
        ''' Save all corpus to transform '''
        if self.prepare_word_embedding:
            if self.train_embedder:
                # self.edge_embedder.prepare(self.edge_args_embedding_data_csv, self.edge_args_embedding_data_csv__cls, self.train_list_name, self.test_list_name, self.flags_keys)
                # self.node_embedder.prepare(self.node_name_embedding_data_csv, self.node_name_embedding_data_csv__cls, self.train_list_name, self.test_list_name)
                self.edge_embedder.prepare(self.edge_args_embedding_data_csv, self.train_list_name, self.test_list_name, self.flags_keys)
                self.node_embedder.prepare(self.node_name_embedding_data_csv, self.train_list_name, self.test_list_name)

            self.edge_embedder.save_corpus(self.edge_args_embedding_data_csv)
            self.node_embedder.save_corpus(self.node_name_embedding_data_csv)
        else:
            # Copy corpus file from emb_trained_path
            if not os.path.exists(self.edge_args_emb_corpus_path+'/corpus.csv'):
                shutil.copy(self.edge_args_emb_trained_path+'/corpus.csv', self.edge_args_emb_corpus_path+'/corpus.csv')
                shutil.copy(self.node_name_emb_trained_path+'/corpus.csv', self.node_name_emb_corpus_path+'/corpus.csv')


        if self.train_embedder:
            ''' Train embedding for edge arguments and node names '''
            self.word_to_ix_edge, word_dict_edge = self.edge_embedder.train()
            self.word_to_ix_node, word_dict_node = self.node_embedder.train()
            ''' Load on this whole corpus '''
            self.edge_embedder.load(load_train_test_set=False)
            self.node_embedder.load(load_train_test_set=False)
        else:
            ''' Load tf-idf '''
            print('\n---------------- [train_and_load_embedding] Load vectorizer')
            self.word_to_ix_edge, word_dict_edge = self.edge_embedder.load()
            self.word_to_ix_node, word_dict_node = self.node_embedder.load()

        for w in word_dict_node:
            # print('w', w)
            if w not in self.word_dict_node:
                self.word_dict_node.append(w)
        for w in word_dict_edge:
            if w not in self.word_dict_edge:
                self.word_dict_edge.append(w)

        if 'other' not in self.word_dict_node:
            self.word_dict_node = ['other'] + self.word_dict_node
        if 'null' not in self.word_dict_edge:
            self.word_dict_edge = ['null'] + self.word_dict_edge


        ''' save vocab '''
        if self.prepend_vocab is True:
            print('[train_and_load_embedding] save vocab')
            with open(self.vocab_path_node, 'w') as f:
                f.write(' '.join(self.word_dict_node))
        if self.prepend_vocab is True:
            print('[train_and_load_embedding] save vocab')
            with open(self.vocab_path_edge, 'w') as f:
                f.write(' '.join(self.word_dict_edge))



    def read_report(self, report_file_path):
        # print('report_file_path', report_file_path)
        with open(report_file_path) as json_file:
            data = json.load(json_file)
            if 'behavior' in data.keys():
                return data['behavior']
            else:
                print('[read_report] No behavior tag found.')
                return None
            # self.behavior = data['behavior']

    def load_data_json(self):
        for key in self.json_data_paths:
                with open(self.json_data_paths[key]) as json_file:
                    print('[load_data_json] Load '+self.json_data_paths[key])
                    self.json_data[key] = json.load(json_file)

    def add_edge_args_embedding_data(self, flags, label, graph_name):
        flags_data = []
        for flag_key in flags:
            if flag_key not in self.flags_keys:
                self.flags_keys.append(flag_key)

            if len(flags[flag_key]) > 0:
                # flag_data = flags[flag_key].replace('|', ' ').lower()
                flag_data = flag_key + ' ' + flags[flag_key].replace('|', ' ').lower()
                flags_data.append(flag_data)
        
        if len(flags_data) > 0:
            if graph_name not in self.edge_args_by_graph:
                self.edge_args_by_graph[graph_name] = flags_data
            else:
                self.edge_args_by_graph[graph_name] += flags_data

    def gen_edge_args_embedding_data(self):
        print('[gen_edge_args_embedding_data] self.edge_args_by_graph', len(self.edge_args_by_graph))
        for graph_name in self.edge_args_by_graph:
            label = graph_name.split('__')[0]
            flags_data_txt = ' '.join(self.edge_args_by_graph[graph_name])
            # self.edge_args_embedding_data_csv__cls[label].append('{} {}'.format(flag_key, flags_data_txt))

            if self.has_label is True:
                self.edge_args_embedding_data_csv.append({'class': self.mapping[label], 'data': flags_data_txt, 'file': graph_name})
                self.edge_args_embedding_data_csv__cls[label].append({'class': self.mapping[label], 'data': flags_data_txt})
            else:
                self.edge_args_embedding_data_csv.append({'class': -1, 'data': flags_data_txt, 'file': graph_name})
                # self.edge_args_embedding_data_csv__cls[label].append({'class': -1, 'data': flags_data_txt})


    # def load_edge_args_embedding_data(self, flags, label):
    #     self.edge_args_embedding_data_csv = pd.read_csv("filename.csv")
    #     self.edge_args_embedding_data_csv__cls[label].append({'class': self.mapping[label], 'data': flags_data_txt})


    def gen_node_name_embedding_data(self):
        # print('\t [gen_node_name_embedding_data] gen_node_name_embedding_data', node_name)
        print('[gen_node_name_embedding_data] self.node_names_by_graph', len(self.node_names_by_graph))
        for graph_name in self.node_names_by_graph:
            label = graph_name.split('__')[0]
            txt = ' '.join(self.node_names_by_graph[graph_name])
            node_name = self.nodename_to_str(txt)

            if self.has_label is True:
                self.node_name_embedding_data_csv.append({'class': self.mapping[label], 'data': node_name})
                # self.node_name_embedding_data_csv__cls[label].append('{}'.format(node_name))
                self.node_name_embedding_data_csv__cls[label].append({'class': self.mapping[label], 'data': node_name})
            else:
                self.node_name_embedding_data_csv.append({'class': -1, 'data': node_name})
                # self.node_name_embedding_data_csv__cls[label].append({'class': -1, 'data': node_name})

    def add_node_name_embedding_data(self, name, label, graph_name):
        if len(name) > 0:
            if graph_name not in self.node_names_by_graph:
                self.node_names_by_graph[graph_name] = [name]
            else:
                self.node_names_by_graph[graph_name] += [name]


    def encode_report(self, behavior, report_name, report_folder):
        """
        Process the data extracted from the report and save to data.json.
        Encode node, edge and save to graphs_dict.
        --------------------------------------
        Get and save the nodes, meta-path
            - report_folder: category of report (graph label)
            - report_name: graph name (graph id)
        Encode node and edge and save to graphs_dict also
        """

        # to debug single file
        # print('[encode_report] report_name', report_name)
        # if report_name != '8f5135eec4dcb808423209163bbd94025ec47f4cb1b20dcf75b1fd56773ac58f.json':
        #     return

        self.current_node_id_of_current_graph = -1
        graph_name = report_folder+'__'+report_name
        print('[encode_report] graph_name', graph_name)
        
        #####################
        # Get all the procs
        #####################
        # print(len(behavior['processes']))
        # print(behavior['processtree'])
        procs = behavior['processes']

        # print('=================================')
        # print(self.pid_to_node)
        # print('=================================\n')

        for proc in procs:
            calls = proc['calls']
            
            # id \t proc_name \t proc_path_severity \t regkey_written_severity \t dll_loaded_severity \t connects_host_severity
            # proc_name = proc['process_name']
            proc_name = proc['process_path']
            proc_info = '{}|{}'.format(graph_name, proc_name)

            if len(calls) > 0:  # this process does have api calls
                ##############################
                # Now loop through all the api calls
                ##############################
                for api in calls:
                    cat = api['category']
                    

                    # if cat in self.allow_cat and api['api'].lower() in self.interesting_apis:
                    if cat in self.allow_cat:

                        if bool(api['flags']) is True: # not empty
                            self.add_edge_args_embedding_data(api['flags'], report_folder, graph_name)

                        __proc_identifier__ = graph_name+'_proc_'+str(proc['pid'])
                        # create parent node (root proc) if not inserted yet
                        if __proc_identifier__ not in self.pid_to_node:
                            self.increase_node()
                            proc_data = {
                                # 'name': proc_name.replace(' ', '^'),
                                'name': 'proc{'+str(proc['pid'])+'}',
                                'pid': proc['pid'],
                                'type': 'proc',
                                'id': self.current_node_id,
                                'id_in_graph': self.current_node_id_of_current_graph,
                                'graph': graph_name,
                                'graph_label': report_folder
                            }
                            # self.json_data['nodes'].append(proc_data)
                            self.json_data['nodes'][proc_data['id']] = proc_data
                            self.add_node_name_embedding_data('proc', report_folder, graph_name)
                            
                            self.pid_to_node[graph_name+'_proc_'+str(proc['pid'])] = proc_data
                            
                        
                        # if api['api'].lower() in self.interesting_apis:
                        #     api_name = api['api'].lower()
                        # else:
                        #     api_name = 'other'
                        api_name = api['api'].lower()
                        
                        api_time = api['time']
                        api_info = '{}|{}'.format(graph_name, api_name)

                        # print(api)

                        if api_name in self.interesting_apis:
                            if cat == 'file': # process API type file
                                self.process_API_file(api, api_info, proc_data, graph_name, report_folder)

                            if cat == 'process': # process API type process
                                self.process_API_process(api, api_info, proc_data, graph_name, report_folder)

                            if cat == 'registry': # process API type registry
                                self.process_API_registry(api, api_info, proc_data, graph_name, report_folder)
                            
                
        # print('\tDone')
    
    def increase_node(self, api_info=None):
        self.current_node_id = self.current_node_id + 1
        self.current_node_id_of_current_graph = self.current_node_id_of_current_graph + 1

        self.api_nodes_existed_id[api_info] = self.current_node_id


    def process_API_process(self, api, api_info, parent_node, graph_name, graph_label):
        api_name = api['api'].lower()
        api_flags = None
        if 'flags' in api and api['flags'] is not None and api['flags'] != '':
            api_flags = api['flags']
        # api_args = 'NULL'
        # if 'arguments' in api and api['arguments'] is not None and api['arguments'] != '':
        #     api_args = getInterestingArg(api['arguments'])


        
        # Check if this api (with the same characteristics (use name only as characteristics)) is called.
        # If not called, create new node
        if api_info not in self.api_nodes_existed_id.keys():
            # create a process api node
            self.increase_node()
            node_api__data = {
                'name': api['api'].lower(),
                'type': 'process_api',
                
                'id': self.current_node_id,
                'id_in_graph': self.current_node_id_of_current_graph,

                'graph': graph_name,
                'graph_label': graph_label
            }
            self.api_nodes_existed[api_info] = node_api__data
        else:
            node_api__data = self.api_nodes_existed[api_info]

        # self.json_data['nodes'].append(node_api__data)
        self.json_data['nodes'][node_api__data['id']] = node_api__data
        self.add_node_name_embedding_data(api['api'].lower(), graph_label, graph_name)

        self.api_nodes_existed_id[api_info] = self.current_node_id

        buffer_length = 0
        if 'buffer' in api['arguments'] and 'length' in api['arguments']:
            buffer_length = api['arguments']['length']

        # if this api has file_handle, then get the file_handle, then find the node correspond with this file_handle, then connect the api_node with the file_handle node
        if 'process_handle' in api['arguments'] and 'process_identifier' in api['arguments'] and api['arguments']['process_identifier'] != 0:

            # create this process API node ONLY WHEN there is a reference from this API to the a process handle (process_handle != 0)
            # node process API data
            # self.increase_node()
            # node_api__data = {
            #     'name': api['api'].lower(),
            #     'type': 'process_api',
                    
            #     'id': self.current_node_id,
            #     'id_in_graph': self.current_node_id_of_current_graph,

            #     'graph': graph_name,
            #     'graph_label': graph_label
            # }
            # self.json_data['nodes'].append(node_api__data)
            # self.add_node_name_embedding_data(api['api'].lower(), graph_label, graph_name)
            # self.api_nodes_existed_id[api_info] = self.current_node_id


            process_identifier = str(api['arguments']['process_identifier'])
            # process_handle = api['arguments']['process_handle']

            # save this node to list pid_to_node first, in case later needs query
            # self.pid_to_node[graph_name+'_'+api['pid']] = node_api__data

            __identifier__ = graph_name+'_proc_'+process_identifier
            if __identifier__ not in self.pid_to_node:
                # if api_name == 'NtOpenProcess':
                # create a process node
                self.increase_node()
                # node file handle data
                node_process__data = {
                        'name': 'proc{'+str(process_identifier)+'}',
                        'type': 'proc',
                        
                        'id': self.current_node_id,
                        'id_in_graph': self.current_node_id_of_current_graph,

                        'graph': graph_name,
                        'graph_label': graph_label,
                }
                # self.json_data['nodes'].append(node_process__data)
                self.json_data['nodes'][node_process__data['id']] = node_process__data
                self.add_node_name_embedding_data('proc', graph_label, graph_name)

                self.pid_to_node[__identifier__] = node_process__data
                    
            # get the node to connect to
            connect_node = self.pid_to_node[__identifier__]
                                    
            # create an edge from parent_node to node_api if there the process_identifier is different from parent_node's pid
            if int(parent_node['pid']) != int(process_identifier):
                self.edge(parent_node, node_api__data, args={'api_flags': api_flags, 'edge_type': 'proc_process'}, graph_name=graph_name, buffer_size=buffer_length)
            else:
                # create edge between this node and connect_node
                if 'Open' in api_name or 'Set' in api_name or 'Write' in api_name or 'Create' in api_name:
                    self.edge(node_api__data, connect_node, args={'api_flags': api_flags, 'edge_type': 'proc_process'}, graph_name=graph_name, buffer_size=buffer_length)
                else:
                    self.edge(connect_node, node_api__data, args={'api_flags': api_flags, 'edge_type': 'proc_process'}, graph_name=graph_name, buffer_size=buffer_length)
        
        # Actually we don't care about those API that do not reference process_identifier to any process_identifier, so just comment these
        else:
            # create edge from this api to proc node (parent_node)
            # but because this is process, this edge is the same with the edge created above (from node_api__data to connect_node)
            self.edge(parent_node, node_api__data, args={'api_flags': api_flags, 'edge_type': 'proc_process'}, graph_name=graph_name, buffer_size=buffer_length)
                

    def process_API_file(self, api, api_info, parent_node, graph_name, graph_label):
        api_name = api['api'].lower()

        api_flags = None
        if 'flags' in api and api['flags'] is not None and api['flags'] != '':
            api_flags = api['flags']

        # Check if this api (with the same characteristics (use name only as characteristics)) is called.
        # If not called, create new node
        if api_info not in self.api_nodes_existed_id.keys():
            # create a process api node
            self.increase_node()
            node_api__data = {
                'name': api['api'].lower(),
                'type': 'process_api',
                        
                'id': self.current_node_id,
                'id_in_graph': self.current_node_id_of_current_graph,

                'graph': graph_name,
                'graph_label': graph_label
            }
            self.api_nodes_existed[api_info] = node_api__data
        else:
            node_api__data = self.api_nodes_existed[api_info]

        # self.json_data['nodes'].append(node_api__data)
        self.json_data['nodes'][node_api__data['id']] = node_api__data
        self.add_node_name_embedding_data(api['api'].lower(), graph_label, graph_name)

        self.api_nodes_existed_id[api_info] = self.current_node_id

        # if this api has file_handle, then get the file_handle, then find the node correspond with this file_handle, then connect the api_node with the file_handle node
        if 'file_handle' in api['arguments']:
            file_handle = api['arguments']['file_handle']

            __identifier__ = graph_name+'_file_'+file_handle
            # if api_name == 'NtCreateFile':
            if __identifier__ not in self.handle_to_node:
                # create a file node
                self.increase_node()
                # node file handle data
                node_file__data = {
                    'name': 'file{'+file_handle+'}',
                    'type': 'file',
                    
                    'id': self.current_node_id,
                    'id_in_graph': self.current_node_id_of_current_graph,

                    'graph': graph_name,
                    'graph_label': graph_label,
                }
                # self.json_data['nodes'].append(node_file__data)
                self.json_data['nodes'][node_file__data['id']] = node_file__data
                self.add_node_name_embedding_data('file', graph_label, graph_name)

                self.handle_to_node[__identifier__] = node_file__data
            
            connect_node = self.handle_to_node[__identifier__]

            buffer_length = 0
            if 'buffer' in api['arguments'] and 'length' in api['arguments']:
                buffer_length = api['arguments']['length']

            # create edge from this api node to connect_node (file handle) (file node)
            if 'Open' in api_name or 'Set' in api_name or 'Write' in api_name or 'Create' in api_name:
                self.edge(node_api__data, connect_node, args={'edge_type': 'proc_file'}, graph_name=graph_name, buffer_size=buffer_length)
            else:
                self.edge(connect_node, node_api__data, args={'edge_type': 'proc_file'}, graph_name=graph_name, buffer_size=buffer_length)

        # create edge from this api to proc node (parent_node)
        self.edge(parent_node, node_api__data, args={'api_flags': api_flags, 'edge_type': 'proc_file'}, graph_name=graph_name)



    def process_API_registry(self, api, api_info, parent_node, graph_name, graph_label):
        api_name = api['api'].lower()

        api_flags = None
        if 'flags' in api and api['flags'] is not None and api['flags'] != '':
            api_flags = api['flags']

        # Check if this api (with the same characteristics (use name only as characteristics)) is called.
        # If not called, create new node
        if api_info not in self.api_nodes_existed_id.keys():
            # create a process api node
            self.increase_node()
            node_api__data = {
                'name': api['api'].lower(),
                'type': 'process_api',
                        
                'id': self.current_node_id,
                'id_in_graph': self.current_node_id_of_current_graph,

                'graph': graph_name,
                'graph_label': graph_label
            }
            self.api_nodes_existed[api_info] = node_api__data
        else:
            node_api__data = self.api_nodes_existed[api_info]
        
            # print('node_api__data', node_api__data)

        # self.json_data['nodes'].append(node_api__data)
        self.json_data['nodes'][node_api__data['id']] = node_api__data
        self.add_node_name_embedding_data(api['api'].lower(), graph_label, graph_name)

        self.api_nodes_existed_id[api_info] = self.current_node_id

        # if this api has key_handle, then get the key_handle, then find the node correspond with this key_handle, then connect the api_node with the key_handle node
        if 'key_handle' in api['arguments']:
            key_handle = api['arguments']['key_handle']

            __identifier__ = graph_name+'_reg_'+key_handle
            # if api_name == 'NtOpenKey':
            if __identifier__ not in self.handle_to_node:
                # create a registry key node
                self.increase_node()
                # node key handle data
                node_reg__data = {
                    'name': 'reg{'+key_handle+'}',
                    'type': 'reg',
                    
                    'id': self.current_node_id,
                    'id_in_graph': self.current_node_id_of_current_graph,

                    'graph': graph_name,
                    'graph_label': graph_label,
                }
                # self.json_data['nodes'].append(node_reg__data)
                self.json_data['nodes'][node_reg__data['id']] = node_reg__data
                self.add_node_name_embedding_data('reg', graph_label, graph_name)

                self.handle_to_node[__identifier__] = node_reg__data
            
            connect_node = self.handle_to_node[__identifier__]

            buffer_length = 0
            if 'buffer' in api['arguments'] and 'length' in api['arguments']:
                buffer_length = api['arguments']['length']

            # create edge from this api node to connect_node (file handle) (file node)
            if 'Open' in api_name or 'Set' in api_name or 'Write' in api_name or 'Create' in api_name:
                self.edge(node_api__data, connect_node, args={'edge_type': 'proc_reg'}, graph_name=graph_name, buffer_size=buffer_length)
            else:
                self.edge(connect_node, node_api__data, args={'edge_type': 'proc_reg'}, graph_name=graph_name, buffer_size=buffer_length)

        # create edge from this api to proc node (parent_node)
        self.edge(parent_node, node_api__data, args={'api_flags': api_flags, 'edge_type': 'proc_reg'}, graph_name=graph_name)



    def edge(self, s, d, args, graph_name, buffer_size=0):
        self.current_edge_id += 1

        # if buffer_size <= 0:
        #     return

        if buffer_size > 0:
            print('[edge] buffer', buffer_size)

        path_data = {
            # 'type': self.path_type_code[args['edge_type']],
            'type': args['edge_type'],
            'args': {},
            'from': s['id'],
            'to': d['id'],
            
            'from_in_graph': s['id_in_graph'],
            'to_in_graph': d['id_in_graph'],
            
            'id': self.current_edge_id,
            'buffer_size': buffer_size,

            'graph': graph_name
        }

        # care only when this api (s and d) is not other (in interesting apis)
        if args is not None and 'api_flags' in args and self.nodename_to_viz(s['name']) != 'other' and self.nodename_to_viz(d['name']) != 'other':
            # print('args', args, "args['api_flags']", args['api_flags'])
            path_data['args'] = args['api_flags']
            
        # self.json_data[args['edge_type']].append(path_data)
        self.json_data[args['edge_type']][path_data['id']] = path_data
        # self.json_data['path'][path_data['id']] = path_data



    def encode_node(self, node):
        """
        Encode node information to node attribute
        ----------------------------
            Calculate node attributes (init features)
            All nodes must have same features space.
        """
        # =======================================
        # Encode node name using Word Embedding
        # =======================================

        # Use name of API to represent each node
        self.nodes_labels = self.nodes_labels + [node['name']]

        ###################
        # Create graph
        ###################
        if node['graph'] not in self.graphs_dict.keys():
            self.graphs_name_to_label[node['graph']] = node['graph_label']
            self.graphs_dict[node['graph']] = DGLGraph(multigraph=True)
            self.graphs_viz[node['graph']] = Digraph(
                name=node['graph_label'], format='png')

        ###################
        # Get features
        ###################

        ndata = {}

        ''' GNN_NODE_TYPES_KEY '''
        node_type_encoded = indices_to_one_hot(
            self.node_type_code[node['type']], out_vec_size=len(self.node_type_code))
        nte_torch = torch.from_numpy(np.array([node_type_encoded])).type(torch.FloatTensor)
        # print('nte_torch', nte_torch)
        ndata[GNN_NODE_TYPES_KEY] = nte_torch

        ''' GNN_NODE_LABELS_KEY '''
        # cbow_node = self.cbow_encode_node_name(self.nodename_to_str(node['name']))
        name_transformed = self.node_embedder.transform(self.nodename_to_str(node['name']))
        cbow_node = torch.tensor(name_transformed).type(torch.FloatTensor)
        ndata[GNN_NODE_LABELS_KEY] = cbow_node.view(1, -1)

        # print('ndata[GNN_NODE_LABELS_KEY]', ndata[GNN_NODE_LABELS_KEY])
        # print('ndata[GNN_NODE_TYPES_KEY]', ndata[GNN_NODE_TYPES_KEY])

        ''' add node with data to graph '''
        self.graphs_dict[node['graph']].add_nodes(1, data=ndata)

        # for visualize
        if node['type'] in ['proc', 'file', 'reg']:
            shape = 'ellipse'
        else:
            shape = 'box' # api node
        self.graphs_viz[node['graph']].node(
            'n{}'.format(node['id_in_graph']), self.nodename_to_viz(node['name']), color=self.node_color[self.node_type_code[node['type']]], shape=shape)

    def encode_edge(self, path):
        """
        Encode edge information to node attribute
        """

        if len(path) <= 0:
            del self.graphs_name_to_label[path['graph']]
            del self.graphs_dict[path['graph']]
            del self.graphs_viz[path['graph']]
            return

        self.edges_labels.append(self.path_type_code[path['type']])
        # self.edges_labels.append(path['type_code'])

        edata = {}

        ''' GNN_EDGE_TYPES_KEY '''
        edge_type_encoded = indices_to_one_hot(
            self.path_type_code[path['type']], out_vec_size=len(self.path_type_code))
        ete_torch = torch.from_numpy(
            np.array([edge_type_encoded])).type(torch.FloatTensor)
        # print('ete_torch', ete_torch)
        edata[GNN_EDGE_TYPES_KEY] = ete_torch

        ''' GNN_EDGE_LABELS_KEY '''
        # cbow_edge = self.cbow_encode_edge_args(args_to_str(path['args']))
        # args_transformed, txt_chosen = self.edge_embedder.transform(args_to_str(path['args']))
        args_transformed = self.edge_embedder.transform(args_to_str(path['args']))
        cbow_edge = torch.tensor(args_transformed).type(torch.FloatTensor)
        edata[GNN_EDGE_LABELS_KEY] = cbow_edge.view(1, -1)
        
        # print("\nargs_to_str(path['args'])", txt_chosen, '||', args_to_str(path['args']))
        # print('edata[GNN_EDGE_TYPES_KEY]', edata[GNN_EDGE_TYPES_KEY])
        # print('edata[GNN_EDGE_LABELS_KEY]', edata[GNN_EDGE_LABELS_KEY])
        # print('edata[GNN_EDGE_LABELS_KEY].shape', edata[GNN_EDGE_LABELS_KEY].shape)

        ''' GNN_EDGE_BUFFER_SIZE_KEY '''
        edata[GNN_EDGE_BUFFER_SIZE_KEY] = torch.Tensor([[path['buffer_size']]])
        # print('edata[GNN_EDGE_LABELS_KEY]', edata[GNN_EDGE_LABELS_KEY])
        # print('edata[GNN_EDGE_BUFFER_SIZE_KEY]', edata[GNN_EDGE_BUFFER_SIZE_KEY])
        # print(path['buffer_size'])
            

        ''' add edge with data to graph '''
        # print("path['graph']", path['graph'])
        # print(self.graphs_dict[path['graph']].number_of_nodes())
        self.graphs_dict[path['graph']].add_edge(
            path['from_in_graph'], path['to_in_graph'], data=edata)
        self.graphs_viz[path['graph']].edge('n{}'.format(
            path['from_in_graph']), 'n{}'.format(path['to_in_graph']))



    def gen_vocab_and_corpus(self):
        # first create dictionary
        if not os.path.isdir(self.vocab_path):
            os.makedirs(self.vocab_path)
        
        if not os.path.exists(self.vocab_path_node):
            self.create_dict_node()
            self.prepend_vocab = True
        if not os.path.exists(self.vocab_path_edge):
            self.create_dict_edge()
            self.prepend_vocab = True

        # read from dict node
        with open(self.vocab_path_node, 'r') as f:
            vocab = f.read().strip()
            self.word_dict_node = vocab.split(' ')
            self.append_dict_node()
            self.word_to_ix_node = {word: i for i,
                               word in enumerate(self.word_dict_node)}
        self.num_token_node = len(self.word_dict_node)

        # read from dict edge
        with open(self.vocab_path_edge, 'r') as f:
            vocab = f.read().strip()
            self.word_dict_edge = vocab.split(' ')
            # print('self.word_dict_edge', self.word_dict_edge)
            self.append_dict_edge()
        


        ''' Edge arguments embedding by TF-IDF '''
        if self.train_embedder:
            self.edge_embedder.train()
            self.node_embedder.train()

        ''' Save all corpus to transform '''
        if self.from_folder:
            self.edge_embedder.save_corpus(self.edge_args_embedding_data_csv)
            self.node_embedder.save_corpus(self.node_name_embedding_data_csv)


        ''' Load tf-idf '''
        self.word_to_ix_edge, self.word_dict_edge = self.edge_embedder.load()
        self.word_to_ix_node, self.word_dict_node = self.node_embedder.load()


    def encode_data(self):
        """
        Encode nodes & edges from data.json
        """

        if 'nodes' in self.json_data.keys():
            n_num = 0
            n_tot = len(self.json_data['nodes'])
            # self.embed_nodes = nn.Embedding(n_tot, 1)
            print('\n[encode_data] encode_node')
            for node_id in self.json_data['nodes']:
                node = self.json_data['nodes'][node_id]
                # print('encode_node ', node)
                self.encode_node(node)
                n_num += 1
                if n_num % 100000 == 0 or n_num == n_tot:
                    print('[encode_data] {}/{}'.format(n_num, n_tot))
        
        for key in self.json_data:
            if key != 'nodes':
                p_num = 0
                p_tot = len(self.json_data[key])
                # self.embed_edges = nn.Embedding(p_tot, 1)
                print('\n[encode_data] encode_edge type '+key)
                for path_id in self.json_data[key]:
                    path = self.json_data[key][path_id]
                    # print('encode_edge ', path)
                    self.encode_edge(path)
                    p_num += 1
                    if p_num % 100000 == 0 or p_num == p_tot:
                        print('[encode_data] {}/{}'.format(p_num, p_tot))

    def create_graphs(self):
        """
        Create graphs from encoded data to feed to network
        """
        # print(self.graphs_name_to_label)
        print('[create_graphs] len(self.graphs_dict)', len(self.graphs_dict))
        ##############################
        # Append to graphs list
        ##############################
        gnum = 0
        for g_name in list(self.graphs_name_to_label.keys()):
            g_label = self.graphs_name_to_label[g_name]
            graph = self.graphs_dict[g_name]

            if not graph.edata:
                del self.graphs_name_to_label[g_name]
                del self.graphs_dict[g_name]

            else:
                n_nodes = graph.number_of_nodes()
                if n_nodes > self.max_n_nodes:
                    self.max_n_nodes = n_nodes

                ######################
                # Normalize edge
                ######################
                # edge_src, edge_dst = graph.edges()

                # edge_dst = list(edge_dst.data.numpy())
                # print('graph.edata ('+g_label+')', graph.edata)
                # # edge_type = list(graph.edata[GNN_EDGE_TYPES_KEY])
                # edge_lbl = list(graph.edata[GNN_EDGE_LABELS_KEY])

                # # print('edge_dst, edge_type', edge_dst, edge_type)
                # # _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True, return_counts=True)
                # _, inverse_index, count = np.unique((edge_dst, edge_lbl), axis=1, return_inverse=True, return_counts=True)
                # degrees = count[inverse_index]
                # edge_norm = np.ones(
                #     len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)
                # graph.edata[GNN_EDGE_NORM] = torch.FloatTensor(edge_norm)

                self.graphs.append(graph)
                self.graphs_names.append(g_name)
                self.graphs_labels.append(g_label)

                # Save this graph to png
                # if gnum < 10:
                # json_file_size = os.path.getsize(self.reports_parent_dir_path+'/'+g_label+'/'+g_name.split('__')[1])
                # if json_file_size // 1000000 <= 250: # 250000000
                if False:
                    # print(graph)
                    # nx.draw(graph.to_networkx(), with_labels=True)
                    # plt.savefig('data/graphs/{}.png'.format(g_name))
                    # print(self.graphs_viz[g_name].source)
                    self.graphs_viz[g_name].render(
                        filename='data/graphviz/{}_edge/{}'.format(os.path.basename(self.reports_parent_dir_path), g_name))
                gnum += 1

        # print(self.graphs)
        
        num_entities = len(set(self.nodes_labels))
        num_rels = len(set(self.edges_labels))

        self.save_pickle_data()

        # Save additional data
        save_pickle(num_entities, os.path.join(self.pickle_folder, N_ENTITIES))
        save_pickle(num_rels, os.path.join(self.pickle_folder, N_RELS))
        save_pickle(self.max_n_nodes, os.path.join(self.pickle_folder, MAX_N_NODES))

        self.data_dortmund_format[N_ENTITIES] = num_entities
        self.data_dortmund_format[N_RELS] = num_rels
        self.data_dortmund_format[MAX_N_NODES] = self.max_n_nodes


    def save_pickle_data(self):
        save_pickle(self.graphs, os.path.join(self.pickle_folder, GRAPH))
        save_pickle(self.graphs_names, os.path.join(self.pickle_folder, GNAMES))
        save_txt(self.graphs_names, os.path.join(self.pickle_folder, GNAMES+'.txt'))

        self.data_dortmund_format = {
            GRAPH: self.graphs,
            GNAMES: self.graphs_names,
        }

        if self.has_label is True:
            if self.mapping_path is None:
                label_set = set(sorted(self.graphs_labels))  # malware: 0, benign: 1
                num_labels = len(label_set)
                self.mapping = dict(zip(label_set, list(range(num_labels))))
                if self.pickle_folder is not None:
                    with open(self.pickle_folder+'/mapping.json', 'w') as f:
                        json.dump(self.mapping, f)
            
            # mapping = self.mapping_labels
            num_labels = len(self.mapping)
            print('[save_pickle_data] num_labels', num_labels)
            print('[save_pickle_data] mapping', self.mapping)
            labels = [self.mapping[label] for label in self.graphs_labels]
            # print('[save_pickle_data] labels', labels)
            # print('[save_pickle_data] label_set', label_set)

            labels_torch = torch.LongTensor(labels)
            print('[save_pickle_data] labels_torch', labels_torch)

            torch.save(labels_torch, os.path.join(self.pickle_folder, LABELS))
            save_pickle(num_labels, os.path.join(self.pickle_folder, N_CLASSES))
            save_pickle(self.graphs_labels, os.path.join(self.pickle_folder, LABELS_TXT))

            self.data_dortmund_format[N_CLASSES] = num_labels
            self.data_dortmund_format[LABELS] = labels_torch
            self.data_dortmund_format[LABELS_TXT] = self.graphs_labels




    def load_from_pickle(self):
        """
        Load data from pickle files
        """
        print('[load_from_pickle]', os.path.join(self.pickle_folder, GRAPH))
        self.data_dortmund_format = {
            GRAPH: load_pickle(os.path.join(self.pickle_folder, GRAPH)),
            GNAMES: load_pickle(os.path.join(self.pickle_folder, GNAMES)),
            N_CLASSES: load_pickle(os.path.join(self.pickle_folder, N_CLASSES)),
            N_ENTITIES: load_pickle(os.path.join(self.pickle_folder, N_ENTITIES)),
            N_RELS: load_pickle(os.path.join(self.pickle_folder, N_RELS)),
            LABELS: torch.load(os.path.join(self.pickle_folder, LABELS)),
            MAX_N_NODES: load_pickle(os.path.join(self.pickle_folder, MAX_N_NODES)),
            LABELS_TXT: load_pickle(os.path.join(self.pickle_folder, LABELS_TXT)),
            # GRAPH_ADJ: torch.load(os.path.join(self.pickle_folder, GRAPH_ADJ))
        }

        return self.data_dortmund_format

    def create_dict_node(self):
        """
        Create dictionary of name and arguments to encode
        """
        print('[create_dict_node]')
        self.word_dict_node.append('other')
        self.append_dict_node()
    
    def create_dict_edge(self):
        """
        Create dictionary of name and arguments to encode
        """
        print('[create_dict_edge]')
        self.word_dict_edge.append('null')
        self.append_dict_edge()
    
    
    def append_dict_node(self):
        ''' use vocab for nodes separately '''
        # if 'nodes' in self.json_data.keys():
        #     nlen = len(self.json_data['nodes'])
        #     print('nlen', nlen)
        #     ncount = 0
        #     for node_id in self.json_data['nodes']:
        #         node = self.json_data['nodes'][node_id]
        #         ncount += 1
        #         if ncount % 10000 == 0 or ncount == nlen:
        #             print('update vocab at node {}/{}'.format(ncount, nlen))
        #         node_name = self.nodename_to_str(node['name'])
        #         if node_name not in self.word_dict_node:
        #             print('\t'+node['name']+' not in word_dict_node', self.prepend_vocab)
        #             if self.prepend_vocab is True:
        #                 self.word_dict_node.append(node_name)
        #             # else:
        #             #     print('\t\tChange node name', node_id, self.json_data['nodes'][node_id]['name'])
        #             #     self.json_data['nodes'][node_id]['name'] = 'other'
        ''' save vocab '''
        if self.prepend_vocab is True:
            print('[append_dict_node] save vocab')
            with open(self.vocab_path_node, 'w') as f:
                f.write(' '.join(self.word_dict_node))


    def append_dict_edge(self):
        # for key in self.json_data:
        #     if key != 'nodes':
        #         plen = len(self.json_data[key])
        #         pcount = 0
        #         for path_id in self.json_data[key]:
        #             path = self.json_data[key][path_id]
        #             pcount += 1
        #             if pcount % 10000 == 0 or pcount == plen:
        #                 print('update vocab at edge type {} {}/{}'.format(key, pcount, plen))
        #             # print('path', path)
        #             txt = args_to_str(path['args'])
        #             # print('txt', txt)
        #             for word in txt.split(' '):
        #                 if len(word) > 0 and word != ' ' and word not in self.word_dict_edge:
        #                     print('\t"'+word+'" not in word_dict_edge')
        #                     # self.word_dict_edge.append(word)

        #                     if self.prepend_vocab is True:
        #                         self.word_dict_edge.append(word)
        #                     # else:
        #                     #     print('\t\tChange edge args', path_id, self.json_data[key][path_id]['args'])
        #                     #     txt = txt.replace(word, 'null')
        #                     #     self.json_data[key][path_id]['args'] = txt

        ''' save vocab '''
        if self.prepend_vocab is True:
            print('[append_dict_edge] save vocab')
            with open(self.vocab_path_edge, 'w') as f:
                f.write(' '.join(self.word_dict_edge))



    def cbow_encode_node_name(self, raw_text):
        data = []
        # target = raw_text
        data.append(raw_text)
        return make_vector(data, self.word_to_ix_node)
        # return make_vector(data, self.word_to_ix_node)


    # def cbow_encode_edge_args(self, raw_text):
    #     data = []
    #     data.append(raw_text)
    #     return make_vector(data, self.word_to_ix_edge)

    def cbow_encode_edge_args(self, raw_text):
        if len(raw_text) == 0 or raw_text == 'null':
            raw_text = 'null null'

        raw_text = raw_text.split(' ')
        # print('\t raw_text', raw_text)
        if len(raw_text) == 1:
            # raw_text.append('null')
            raw_text = ['null', raw_text[0]]
        
        data = []
        i = 1
        while i < len(raw_text):
            target = raw_text[i]
            context = [raw_text[i - 1], target]
            data.append((context, target))
            i += 2
        # print('data', data)
        return make_context_vector(data[0][0], self.word_to_ix_edge)


    def nodename_to_viz(self, txt):
        txt = txt.lower().strip()

        # print('self.use_interesting_apis', self.use_interesting_apis)
        if self.use_interesting_apis is False:
            return txt
        
        if txt.split('{')[0] not in self.interesting_apis:
            txt = 'other{'+txt+'}'
        return txt

    def nodename_to_str(self, txt):
        txt = txt.lower().strip()

        if self.use_interesting_apis is False:
            return txt.split('{')[0]

        if txt.split('{')[0] not in self.interesting_apis:
            return 'other'

        return txt.split('{')[0]


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass


def make_vector(words, word_to_ix):
    idxs = [word_to_ix[w] for w in words]
    return torch.tensor(idxs)
    # return torch.tensor([idxs])


def make_context_vector(context, word_to_ix):
    # print('context', context)
    idxs = []
    for w in context:
        if len(w) == 0:
            v = 0.0
        else:
            w = w.lower()
            v = word_to_ix[w] if w in word_to_ix else 0.0
        idxs.append(v)
    # idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs)
    # return torch.tensor([idxs])


def args_to_str(args_):
    # get flags values only. ignore keys
    # print('args_', args_)
    arr_val = []
    for key in args_:
        values = args_[key].split('|')
        arr_val += values
    arr_val = [v.strip().lower() for v in arr_val]
    return ' '.join(arr_val)

    # str_ = str(args_).lower()
    # str_ = str_.replace('{', '').replace('}', '').replace('\'', '').replace(
    #     '"', '').replace(':', ' ').replace(',', ' ').replace('|', ' ').replace('  ', ' ')
    # return str_


