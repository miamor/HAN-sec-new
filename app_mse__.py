import time
import numpy as np
import dgl
import torch
from torch.utils.data import DataLoader
import random
import json

from utils.early_stopping import EarlyStopping
from utils.io import load_checkpoint
from utils.utils import label_encode_onehot, indices_to_one_hot

from utils.constants import *
from models.model import Model
# from __save_results.gat_nw__8379__1111__cuckoo_ADung__noiapi__vocablower_noiapi_full__tfidf.model import Model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
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

    def __init__(self, data, model_config, learning_config, pretrained_weight, early_stopping=True, patience=100, json_path=None, pickle_folder=None, vocab_path=None, mapping_path=None, odir=None, model_src_path=None):
        if model_src_path is not None:
            sys.path.insert(0, model_src_path+'model.py')
            print('*** model_src_path', model_src_path)
            from model_edgnn import Model
        else:
            from models.model import Model

        self.data = data
        self.model_config = model_config
        # max length of a sequence (max nodes among graphs)
        self.learning_config = learning_config
        self.pretrained_weight = pretrained_weight
        self.is_cuda = learning_config['cuda']

        # with open(vocab_path+'/../mapping.json', 'r') as f:
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)

        self.labels = self.data[LABELS]
        self.graphs_names = self.data[GNAMES]

        self.data_graph = self.data[GRAPH]

        # save nid and eid to nodes & edges
        # print('self.data_graph[0]', self.data_graph[0])
        if 'nid' not in self.data_graph[0].ndata:
        # if True:
            for k,g in enumerate(self.data_graph):
                g = self.write_nid_eid(g)
                self.data_graph[k] = g
            # print('self.data_graph', self.data_graph)
        save_pickle(self.data_graph, os.path.join(pickle_folder, GRAPH))


        data_nclasses = self.data[N_CLASSES]
        if N_RELS in self.data:
            data_nrels = self.data[N_RELS]
        else:
            data_nrels = None
            
        if N_ENTITIES in self.data:
            data_nentities = self.data[N_ENTITIES]
        else:
            data_nentities = None

        self.model = Model(g=self.data_graph[0],
                           config_params=model_config,
                           n_classes=data_nclasses,
                           n_rels=data_nrels,
                           n_entities=data_nentities,
                           is_cuda=self.is_cuda,
                           batch_size=1,
                           json_path=json_path,
                           vocab_path=vocab_path,
                           model_src_path=model_src_path)

        if self.is_cuda is True:
            # self.model.cuda()
            print('Use cuda')
            self.model.to(torch.device('cuda'))


        print('*** Model parameters ***')
        pp=0
        for p in list(self.model.parameters()):
            nn=1
            for s in list(p.size()):
                # print('p', p)
                print('\t s, nn, nn*s', s, nn, nn*s)
                nn = nn*s
            pp += nn
        print('Total params', pp)


        if early_stopping:
            self.early_stopping = EarlyStopping(patience=patience, verbose=True)
            
        # Output folder to save train / test data
        if odir is None:
            odir = 'output/'+time.strftime("%Y-%m-%d_%H-%M-%S")
        self.odir = odir


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


    def train(self, save_path='', k_fold=10, train_list_file=None, test_list_file=None):
        if self.pretrained_weight is not None:
            self.model = load_checkpoint(self.model, self.pretrained_weight, self.is_cuda)
        save_dir = save_path.split('/checkpoint')[0]

        loss_fcn = torch.nn.MSELoss()

        # initialize graphs
        self.accuracies = np.zeros(k_fold)
        graphs = self.data[GRAPH]                 # load all the graphs

        # debug purposes: reshuffle all the data before the splitting
        random_indices = list(range(len(graphs)))
        random.shuffle(random_indices)
        graphs = [graphs[i] for i in random_indices]
        labels = self.labels[random_indices]
        graphs_names = [self.graphs_names[i] for i in random_indices]


        split_train_test = True if train_list_file is None and test_list_file is None else False 
        print('split_train_test', split_train_test)

        if split_train_test is True:
            print('train_list_file', train_list_file)
            print('test_list_file', test_list_file)
            #############################
            # Create new train/test set
            # Split train and test
            #############################
            train_size = int(self.TRAIN_SIZE * len(graphs))
            g_train = graphs[:train_size]
            l_train = labels[:train_size]
            n_train = graphs_names[:train_size]

            g_test = graphs[train_size:]
            l_test = labels[train_size:]
            n_test = graphs_names[train_size:]
            
        else:
            #############################
            # Load train and test graphs from list
            #############################
            train_files = []
            test_files = []
            g_train = []
            l_train = []
            n_train = []
            g_test = []
            l_test = []
            n_test = []
            with open(train_list_file, 'r') as f:
                train_files = [l.strip() for l in f.readlines()]
            with open(test_list_file, 'r') as f:
                test_files = [l.strip() for l in f.readlines()]
            
            for i in range(len(labels)):
                graph_jsonpath = graphs_names[i]
                # print(graph_jsonpath)
                if graph_jsonpath in train_files:
                    g_train.append(graphs[i])
                    l_train.append(labels[i])
                    n_train.append(graphs_names[i])
                if graph_jsonpath in test_files:
                    g_test.append(graphs[i])
                    l_test.append(labels[i])
                    n_test.append(graphs_names[i])

            l_train = torch.Tensor(l_train).type(torch.LongTensor)
            l_test = torch.Tensor(l_test).type(torch.LongTensor)
            if self.is_cuda is True:
                l_train = l_train.cuda()
                l_test = l_test.cuda()


        # print('len g_train', len(g_train))
        # print('g_train', g_train)
        

        if not os.path.isdir(self.odir):
            os.makedirs(self.odir)
        save_pickle(g_train, os.path.join(self.odir, 'train'))
        save_pickle(l_train, os.path.join(self.odir, 'train_labels'))
        save_pickle(g_test, os.path.join(self.odir, 'test'))
        save_pickle(l_test, os.path.join(self.odir, 'test_labels'))

        # save graph name list to txt file
        save_txt(n_train, os.path.join(self.odir, 'train_list.txt'))
        save_txt(n_test, os.path.join(self.odir, 'test_list.txt'))


        K = k_fold
        for k in range(K):                  # K-fold cross validation

            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_config['lr'],
                                         weight_decay=self.learning_config['weight_decay'])

            start = int(len(g_train)/K) * k
            end = int(len(g_train)/K) * (k+1)
            print('\n\n\nProcess new k='+str(k)+' | '+str(start)+'-'+str(end))

            # testing batch
            val_batch_graphs = g_train[start:end]
            val_batch_labels = l_train[start:end]
            val_batch = dgl.batch(val_batch_graphs)

            # training batch
            train_batch_graphs = g_train[:start] + g_train[end:]
            train_batch_labels = l_train[list(
                range(0, start)) + list(range(end+1, len(g_train)))]
            train_batch_samples = list(
                map(list, zip(train_batch_graphs, train_batch_labels)))
            train_batches = DataLoader(train_batch_samples,
                                          batch_size=self.learning_config['batch_size'],
                                          shuffle=True,
                                          collate_fn=collate)

            print('train_batches size: ', len(train_batches))
            print('train_batch_graphs size: ', len(train_batch_graphs))
            print('val_batch_graphs size: ', len(val_batch_graphs))
            print('train_batches', train_batches)
            print('val_batch_labels', val_batch_labels)
            
            dur = []
            for epoch in range(self.learning_config['epochs']):
                self.model.train()
                if epoch >= 3:
                    t0 = time.time()
                losses = []
                training_accuracies = []
                for iter_idx, (bg, label) in enumerate(train_batches):
                    logits = self.model(bg)
                    label = label.float()
                    if self.learning_config['cuda']:
                        label = label.cuda()
                    loss = loss_fcn(logits, label)
                    losses.append(loss.item())
                    print('~~~~ logits', logits)
                    print('~~~~ label', label)

                    t = torch.Tensor([0.5]) # threshold for classifying as malware
                    if self.is_cuda:
                        t = t.cuda()
                    indices = (logits > t).float() * 1
                    print('~~~~ indices', indices)

                    correct = torch.sum(indices == label)
                    training_accuracies.append(
                        correct.item() * 1.0 / len(label))

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                val_acc, val_loss, _ = self.model.eval_graph_classification(val_batch_labels, val_batch)
                print("Epoch {:05d} | Time(s) {:.4f} | train_acc {:.4f} | train_loss {:.4f} | val_acc {:.4f} | val_loss {:.4f}".format(
                    epoch, np.mean(dur) if dur else 0, np.mean(training_accuracies), np.mean(losses), val_acc, val_loss))

                is_better = self.early_stopping(val_loss, self.model, save_path)
                if is_better:
                    self.accuracies[k] = val_acc

                if self.early_stopping.early_stop:
                    # Print model's state_dict
                    print("*** Model's state_dict:")
                    for param_tensor in self.model.state_dict():
                        print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

                    # Print optimizer's state_dict
                    print("*** Optimizer's state_dict:")
                    for var_name in optimizer.state_dict():
                        print(var_name, "\t", optimizer.state_dict()[var_name])

                    # Save state dict
                    torch.save(self.model.state_dict(), save_dir+'/model_state.pt')
                    # Save model
                    # torch.save({
                    #     'epoch': epoch,
                    #     'model_state_dict': self.model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'val_loss': val_loss,
                    # }, save_dir+'/saved')


                    print("Early stopping")
                    break

            self.early_stopping.reset()

    def test(self, model_path=''):
        print('Test model')
        
        try:
            print('*** Load pre-trained model '+model_path+' ***')
            self.model = load_checkpoint(self.model, model_path, self.is_cuda)
        except ValueError as e:
            print('Error while loading the model.', e)

        print('\nTest all')
        # acc = np.mean(self.accuracies)
        # acc = self.accuracies
        graphs = self.data[GRAPH]
        labels = self.labels
        self.run_test(graphs, labels)
                    
        print('\nTest on train graphs')
        graphs = load_pickle(os.path.join(self.odir, 'train'))
        labels = load_pickle(os.path.join(self.odir, 'train_labels'))
        self.run_test(graphs, labels)

        print('\nTest on test graphs')
        graphs = load_pickle(os.path.join(self.odir, 'test'))
        labels = load_pickle(os.path.join(self.odir, 'test_labels'))
        self.run_test(graphs, labels)


    def test_on_data(self, model_path=''):
        print('Test model')
        
        try:
            print('*** Load pre-trained model '+model_path+' ***')
            self.model = load_checkpoint(self.model, model_path, self.is_cuda)
        except ValueError as e:
            print('Error while loading the model.', e)

        print('\nTest on data')
        # acc = np.mean(self.accuracies)
        # acc = self.accuracies
        graphs = self.data[GRAPH]
        labels = self.labels

        self.run_test(graphs, labels)
        # batch_size = 1024
        # batch_num = len(graphs) // batch_size
        # print('batch_num', batch_num)
        # for batch in range(batch_num):
        #     start = (batch)*batch_size
        #     end = (batch+1)*batch_size
        #     graphs = graphs[start:end]
        #     print(batch, len(graphs))
        #     self.run_test(graphs, labels)


    def run_test(self, graphs, labels):
        batches = dgl.batch(graphs)
        acc, _, logits = self.model.eval_graph_classification(labels, batches)
        
        t = torch.Tensor([0.5]) # threshold for classifying as malware
        if self.is_cuda:
            t = t.cuda()
        indices = (logits > t).float() * 1

        labels = labels.cpu()
        indices = indices.cpu()
        # print('labels', labels)
        # print('indices', indices)
        # labels_txt = ['malware', 'benign']
            
        cm = confusion_matrix(y_true=labels, y_pred=indices)
        print(cm)
        print('Total samples', len(labels))
        
        if len(self.mapping) == 2:
            lbl_mal = self.mapping['malware']
            lbl_bng = self.mapping['benign']
            n_mal = (labels == lbl_mal).sum().item()
            n_bgn = (labels == lbl_bng).sum().item()
            tpr = cm[lbl_mal][lbl_mal]/n_mal * 100 # actual malware that is correctly detected as malware
            far = cm[lbl_bng][lbl_mal]/n_bgn * 100  # benign that is incorrectly labeled as malware
            print('TPR', tpr)
            print('FAR', far)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(cm)
        # plt.title('Confusion matrix of the classifier')
        # fig.colorbar(cax)
        # # ax.set_xticklabels([''] + labels)
        # # ax.set_yticklabels([''] + labels)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()


        print("Accuracy {:.4f}".format(acc))
        
        # acc = np.mean(self.accuracies)

        return acc
