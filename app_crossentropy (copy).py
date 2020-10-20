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
# from models.model import Model
# from __save_results.gat_nw__8379__1111__cuckoo_ADung__noiapi__vocablower_noiapi_full__tfidf.model import Model

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

    def __init__(self, data, model_config, learning_config, pretrained_weight, early_stopping=True, patience=100, json_path=None, pickle_folder=None, vocab_path=None, mapping_path=None, odir=None, model_src_path=None, gdot_path=None):
        if model_src_path is not None:
            sys.path.insert(0, model_src_path)
            print('*** [app][__init__] model_src_path', model_src_path)
            from model_edgnn_o import Model
        else:
            from models.model_edgnn_o import Model
        
        print('*** [app][__init__] gdot_path', gdot_path)

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
        # if 'nid' not in self.data_graph[0].ndata:
        # # if True:
        #     for k,g in enumerate(self.data_graph):
        #         g = self.write_nid_eid(g)
        #         self.data_graph[k] = g
        #     # print('self.data_graph', self.data_graph)
        # save_pickle(self.data_graph, os.path.join(pickle_folder, GRAPH))


        self.data_nclasses = self.data[N_CLASSES]
        if N_RELS in self.data:
            self.data_nrels = self.data[N_RELS]
        else:
            self.data_nrels = None
            
        if N_ENTITIES in self.data:
            self.data_nentities = self.data[N_ENTITIES]
        else:
            self.data_nentities = None

        self.ModelObj = Model
        self.model_src_path = model_src_path

        self.model = self.ModelObj(g=self.data_graph[0],
                           config_params=self.model_config,
                           n_classes=self.data_nclasses,
                           n_rels=self.data_nrels,
                           n_entities=self.data_nentities,
                           is_cuda=self.is_cuda,
                           batch_size=1,
                        #    json_path=json_path,
                        #    vocab_path=vocab_path,
                           model_src_path=model_src_path)        

        if self.is_cuda is True:
            print('[app][__init__] Convert model to use cuda')
            self.model = self.model.cuda()
            # self.model = self.model.to(torch.device('cuda:{}'.format(self.learning_config['gpu'])))

        print('>>> [app][__init__] self.model', self.model)
        print('>>> [app][__init__] Check if model use cuda', next(self.model.parameters()).is_cuda)


        # print('*** [app][__init__] Model parameters ***')
        # pp=0
        # for p in list(self.model.parameters()):
        #     nn=1
        #     for s in list(p.size()):
        #         # print('p', p)
        #         print('\t s, nn, nn*s', s, nn, nn*s)
        #         nn = nn*s
        #     pp += nn
        # print('[app][__init__] Total params', pp)


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

        loss_fcn = torch.nn.CrossEntropyLoss()

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
        print('[app][train] split_train_test', split_train_test)

        '''
        if split_train_test is True:
            print('[app][train] train_list_file', train_list_file)
            print('[app][train] test_list_file', test_list_file)
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
        '''

        print('[app][train] len labels', len(labels))
        print('[app][train] len g_train', len(g_train))
        # print('[app][train] g_train', g_train)


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
        for k in range(K):
            self.model = self.ModelObj(g=self.data_graph[0],
                            config_params=self.model_config,
                            n_classes=self.data_nclasses,
                            n_rels=self.data_nrels,
                            n_entities=self.data_nentities,
                            is_cuda=self.is_cuda,
                            batch_size=1,
                            model_src_path=self.model_src_path)
        

            print('*** [app][__init__] Model layers ***')
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print('\t', name, param.data.type())

            print('>>> [app][__init__] self.model.fc.weight.type', self.model.fc.weight.type())
        

            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_config['lr'],
                                         weight_decay=self.learning_config['weight_decay'])

            start = int(len(g_train)/K) * k
            end = int(len(g_train)/K) * (k+1)
            print('\n\n\n[app][train] Process new k='+str(k)+' | '+str(start)+'-'+str(end))

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

            # testing batch
            val_batch_graphs = g_train[start:end]
            val_batch_labels = l_train[start:end]
            # print('[app][train] val_batch_graphs', val_batch_graphs)
            print('[app][train] len val_batch_graphs', len(val_batch_graphs))
            print('[app][train] val_batch_graphs[0].number_of_nodes()', val_batch_graphs[0].number_of_nodes())
            print('[app][train] val_batch_graphs[-1].number_of_nodes()', val_batch_graphs[-1].number_of_nodes())
            val_batch = dgl.batch(val_batch_graphs)

            print('[app][train] train_batches size: ', len(train_batches))
            print('[app][train] train_batch_graphs size: ', len(train_batch_graphs))
            print('[app][train] val_batch_graphs size: ', len(val_batch_graphs))
            print('[app][train] train_batches', train_batches)
            print('[app][train] val_batch_labels', val_batch_labels)
            
            dur = []
            for epoch in range(self.learning_config['epochs']):
                self.model.train()
                if epoch >= 3:
                    t0 = time.time()
                losses = []
                training_accuracies = []
                for iter_idx, (bg, label) in enumerate(train_batches):
                    # print('~~~ [app][train] bg', bg)
                    logits = self.model(bg)
                    if self.learning_config['cuda']:
                        label = label.cuda()
                    loss = loss_fcn(logits, label)
                    losses.append(loss.item())
                    _, indices = torch.max(logits, dim=1)
                    # print('~~~~ logits', logits)
                    # print('------------------')
                    print('\t [app][train] indices', indices)
                    # print('\t label', label)
                    correct = torch.sum(indices == label)
                    training_accuracies.append(
                        correct.item() * 1.0 / len(label))

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # loss.backward()
                    optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                val_acc, val_loss, _ = self.model.eval_graph_classification(val_batch_labels, val_batch)
                print("[app][train] Epoch {:05d} | Time(s) {:.4f} | train_acc {:.4f} | train_loss {:.4f} | val_acc {:.4f} | val_loss {:.4f}".format(
                    epoch, np.mean(dur) if dur else 0, np.mean(training_accuracies), np.mean(losses), val_acc, val_loss))

                is_better = self.early_stopping(val_loss, self.model, save_path)
                if is_better:
                    self.accuracies[k] = val_acc

                if self.early_stopping.early_stop:
                    # Print model's state_dict
                    # print("*** Model's state_dict:")
                    # for param_tensor in self.model.state_dict():
                    #     print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

                    # # Print optimizer's state_dict
                    # print("*** Optimizer's state_dict:")
                    # for var_name in optimizer.state_dict():
                    #     print(var_name, "\t", optimizer.state_dict()[var_name])

                    # Save state dict
                    # torch.save(self.model.state_dict(), save_dir+'/model_state.pt')

                    # Save model
                    # torch.save({
                    #     'epoch': epoch,
                    #     'model_state_dict': self.model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'val_loss': val_loss,
                    # }, save_dir+'/saved')

                    print("[app][train] Early stopping")
                    break

            self.early_stopping.reset()

    def test(self, model_path=''):
        print('[app][test] Test model')
        
        try:
            print('*** [app][test] Load pre-trained model '+model_path+' ***')
            self.model = load_checkpoint(self.model, model_path, self.is_cuda)
        except ValueError as e:
            print('[app][test] Error while loading the model.', e)
        
        self.save_traintest()

        # print('\n[app][test] Test all')
        # # acc = np.mean(self.accuracies)
        # # acc = self.accuracies
        # graphs = self.data[GRAPH]
        # labels = self.labels
        # self.run_test(graphs, labels)
                    
        graphs = load_pickle(os.path.join(self.odir, 'train'))
        labels = load_pickle(os.path.join(self.odir, 'train_labels'))
        print('\n[app][test] Test on train graphs ({})'.format(len(labels)), os.path.join(self.odir, 'train'))
        self.run_test_fold(graphs, labels, fold=300)

        graphs = load_pickle(os.path.join(self.odir, 'test'))
        labels = load_pickle(os.path.join(self.odir, 'test_labels'))
        print('\n[app][test] Test on test graphs ({})'.format(len(labels)), os.path.join(self.odir, 'test'))
        self.run_test_fold(graphs, labels, fold=150)


    def test_on_data(self, model_path=''):
        print('[app][test_on_data] Test model')
        
        try:
            print('*** [app][test_on_data] Load pre-trained model '+model_path+' ***')
            self.model = load_checkpoint(self.model, model_path, self.is_cuda)
        except ValueError as e:
            print('Error while loading the model.', e)

        print('\n[app][test_on_data] Test on data')
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



    def save_traintest(self):
        graphs = self.data[GRAPH] # load all the graphs
        # labels = self.labels
        # graphs_names = self.graphs_names
        # debug purposes: reshuffle all the data before the splitting
        random_indices = list(range(len(graphs)))
        random.shuffle(random_indices)
        graphs = [graphs[i] for i in random_indices]
        labels = self.labels[random_indices]
        graphs_names = [self.graphs_names[i] for i in random_indices]

        if True:
            train_list_file = '/media/tunguyen/TuTu_Passport/MTAAV/HAN-sec-new/__save_results/reverse__TuTu__vocabtutu__iapi__tfidf__topk=3/9691/train_list.txt'
            test_list_file = '/media/tunguyen/TuTu_Passport/MTAAV/HAN-sec-new/__save_results/reverse__TuTu__vocabtutu__iapi__tfidf__topk=3/9691/test_list.txt'

            train_list_file = '/media/tunguyen/TuTu_Passport/MTAAV/HAN-sec-new/data/TuTu_train_list.txt'
            test_list_file = '/media/tunguyen/TuTu_Passport/MTAAV/HAN-sec-new/data/TuTu_test_list.txt'

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


        print('[app][save_traintest] len labels', len(labels))
        print('[app][save_traintest] len l_test', len(l_test))
        print('[app][save_traintest] len l_train', len(l_train))
        tot_bgn = (labels == self.mapping['benign']).sum().item()
        tot_mal = (labels == self.mapping['malware']).sum().item()
        print('[app][save_traintest] tot_bgn', tot_bgn, 'tot_mal', tot_mal)


        if not os.path.isdir(self.odir):
            os.makedirs(self.odir)
        save_pickle(g_train, os.path.join(self.odir, 'train'))
        save_pickle(l_train, os.path.join(self.odir, 'train_labels'))
        save_pickle(g_test, os.path.join(self.odir, 'test'))
        save_pickle(l_test, os.path.join(self.odir, 'test_labels'))


    def run_test_fold(self, graphs, labels, fold=5):
        num_g = len(labels)
        num_g_per_fold = num_g/fold
        cm_all = np.zeros((len(self.mapping), len(self.mapping)))
        # tot_far = 0
        # tot_tpr = 0
        for i in range(fold):
            start_idx = int(i*num_g_per_fold)
            end_idx = int((i+1)*num_g_per_fold)
            print('* [app][test] Test from {} to {} (total={})'.format(start_idx, end_idx, end_idx-start_idx))
            G = graphs[start_idx:end_idx]
            lbls = labels[start_idx:end_idx]
            acc, cm = self.run_test(G, lbls)
            # print('\t ~~ cm', cm)
            cm_all += cm - np.array([[1,0],[0,1]])
            # if cm.shape[0] == 2:
            # tot_far += cm[lbl_bng][lbl_mal]
        
        print(' >> [app][run_test] All FOLD: cm_all', cm_all)

        if len(self.mapping) == 2:
            labels_cpu = labels.cpu()
            lbl_mal = self.mapping['malware']
            lbl_bng = self.mapping['benign']
            n_mal = (labels_cpu == lbl_mal).sum().item()
            n_bgn = (labels_cpu == lbl_bng).sum().item()
            tpr = (cm_all[lbl_mal][lbl_mal]/n_mal * 100).item() # actual malware that is correctly detected as malware
            far = (cm_all[lbl_bng][lbl_mal]/n_bgn * 100).item()  # benign that is incorrectly labeled as malware
            print(' >> [app][run_test] All FOLD: TPR', tpr, 'n_mal', n_mal, ' ||  FAR', far, 'n_bgn', n_bgn)
            total_samples = len(labels)
            total_correct = cm_all[lbl_mal][lbl_mal] + cm_all[lbl_bng][lbl_bng]
            acc_all = (total_correct/total_samples * 100).item()
            print(' >> [app][run_test] All FOLD: Acc', acc_all, '  Total samples', total_samples)


    def run_test(self, graphs, labels):
        batches = dgl.batch(graphs)
        acc, _, logits = self.model.eval_graph_classification(labels, batches)
        _, indices = torch.max(logits, dim=1)
        labels_cpu = labels.cpu()
        indices_cpu = indices.cpu()
        # print('\t [run_test] labels', labels)
        # print('\t [run_test] indices', indices)
        # labels_txt = ['malware', 'benign']
            
        # print('\t [app][run_test] Total samples', len(labels_cpu))

        # prepend this to make sure cm shape is always (2,2)
        labels_cpu = torch.cat((labels_cpu, torch.tensor([0,1])), 0)
        indices_cpu = torch.cat((indices_cpu, torch.tensor([0,1])), 0)

        cm = confusion_matrix(y_true=labels_cpu, y_pred=indices_cpu)
        C = cm / cm.astype(np.float).sum(axis=1)
        # print('\t [app][run_test] confusion_matrix:', cm)
        
        # if len(self.mapping) == 2:
        #     lbl_mal = self.mapping['malware']
        #     lbl_bng = self.mapping['benign']
        #     n_mal = (labels_cpu == lbl_mal).sum().item()
        #     n_bgn = (labels_cpu == lbl_bng).sum().item()
        #     tpr = cm[lbl_mal][lbl_mal]/n_mal * 100 # actual malware that is correctly detected as malware
        #     far = cm[lbl_bng][lbl_mal]/n_bgn * 100  # benign that is incorrectly labeled as malware
        #     print('\t [app][run_test] TPR', tpr, ' ||  FAR', far, 'n_bgn', n_bgn)
        #     # print('\t [app][run_test] FAR', far, 'n_bgn', n_bgn)

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


        print("\t [app][run_test] Accuracy {:.4f}".format(acc))
        
        # acc = np.mean(self.accuracies)

        return acc, cm
