import argparse
import numpy as np

import torch
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # specify which GPU(s) to be used
# from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.regularizers import l2
# from keras.layers import Input, Dropout

# from keras.layers import Dense, Embedding, LSTM, Bidirectional, Activation, Flatten, Conv1D, TimeDistributed, Add, RepeatVector, Permute, Multiply, GRU

# from models import GraphAttention

from utils.prep_data_n import PrepareData

from utils.inits import to_cuda
from utils.io import print_graph_stats, read_params, create_default_path, remove_model

from app import App

from utils.constants import *
import time
import shutil


def load_dataset(args, cuda):
    prep_data = PrepareData(args)
    data = prep_data.load_data(from_folder=args['from_report_folder'],
                               from_json=args['from_data_json'],
                               from_pickle=args['from_pickle'])
    data = to_cuda(data) if cuda is True else data
    return data


def run_app(args, data, cuda):
    # print_graph_stats(data[GRAPH])

    print('*** Load model from', args['config_fpath'])

    ###########################
    # 1. Training
    ###########################
    if args['action'] == "train":
        now = time.strftime("%Y-%m-%d_%H-%M-%S")

        model_config = args['model_configs']

        odir = 'output/'+now
        # default_path = create_default_path(odir+'/checkpoints')
        default_path = create_default_path(odir)
        print('\n*** Set default saving/loading path to:', default_path)

        learning_config = {'lr': args['lr'], 'epochs': args['epochs'],
                           'weight_decay': args['weight_decay'], 'batch_size': args['batch_size'], 'cuda': cuda}
        app = App(data, model_config=model_config,
                  learning_config=learning_config,
                  pretrained_weight=args['checkpoint_file'], early_stopping=True, patience=20, 
                  json_path=args['input_data_folder'], pickle_folder=args['input_pickle_folder'], vocab_path=args['vocab_path'],
                  mapping_path=args['mapping_path'], 
                  odir=odir,
                  model_src_path=None)
        print('\n*** Start training ***\n')
        ''' save config to output '''
        fconfig_name = args['config_fpath'].split('/')[-1].split('.json')[0]
        shutil.copy(src=args['config_fpath'], dst='{}/{}.json'.format(odir, fconfig_name))
        shutil.copy(src=args['config_fpath'], dst='{}/{}_test_data.json'.format(odir, fconfig_name))
        shutil.copy(src='utils/prep_data.py', dst=odir+'/prep_data.py')
        shutil.copy(src='utils/prep_data_n.py', dst=odir+'/prep_data_n.py')
        shutil.copy(src='utils/word_embedding.py', dst=odir+'/word_embedding.py')
        shutil.copy(src='models/model.py', dst=odir+'/model.py')
        files = ['gat_w', 'gat_nw', 'gat_nw_ns', 'edgnn']
        for fo in files:
            shutil.copy(src='models/model_{}.py'.format(fo), dst=odir+'/model_{}.py'.format(fo))
            shutil.copy(src='models/layers/{}.py'.format(fo), dst=odir+'/{}.py'.format(fo))
        shutil.copy(src='main.py', dst=odir+'/main.py')
        shutil.copy(src='app.py', dst=odir+'/app.py')
        ''' train '''
        app.train(default_path, k_fold=args['k_fold'], train_list_file=args['train_list_file'], test_list_file=args['test_list_file'])
        app.test(default_path)
        # remove_model(default_path)

    ###########################
    # 2. Testing
    ###########################
    # if args['action'] == "test" and args['checkpoint_file'] is not None:
    if args['action'] == "test":
        print('\n*** Start testing ***\n')
        learning_config = {'cuda': cuda}
        # odir = 'output/2020-01-14_15-04-01'
        # odir = args['out_dir']

        # update args
        # config_fpath = odir+'/config_edGNN_graph_class.json'
        # print('*** Update config from', config_fpath)
        # args = read_params(config_fpath, verbose=True)

        fconfig_name = args['config_fpath'].split('/')[-1]
        odir = args['config_fpath'].split(fconfig_name)[0]

        model_config = args['model_configs']

        args['checkpoint_file'] = odir+'/checkpoint'


        app = App(data, model_config=model_config, learning_config=learning_config,
                  pretrained_weight=args['checkpoint_file'], early_stopping=True, patience=20, 
                  json_path=args['input_data_folder'], pickle_folder=args['input_pickle_folder'], vocab_path=args['vocab_path'],
                  mapping_path=args['mapping_path'], odir=odir,
                  model_src_path=odir)
        app.test(args['checkpoint_file'])


def run_app_2(args, data, cuda):
    # config_params = read_params(args.config_fpath, verbose=True)
    # odir = args['out_dir']

    # update args
    # config_fpath = odir+'/config_edGNN_graph_class_test_data.json'
    # print('*** Update config from', config_fpath)
    # args = read_params(config_fpath, verbose=True)

    fconfig_name = args['config_fpath'].split('/')[-1]
    odir = args['config_fpath'].split(fconfig_name)[0]

    model_config = args['model_configs']

    args['checkpoint_file'] = odir+'/checkpoint'


    print('\n*** Start testing ***\n')
    learning_config = {'cuda': cuda}

    app = App(data, model_config=model_config, learning_config=learning_config,
              pretrained_weight=args['checkpoint_file'], early_stopping=True, patience=20, 
              json_path=args['input_data_folder'], pickle_folder=args['input_pickle_folder'], vocab_path=args['vocab_path'],
              mapping_path=args['mapping_path'],
              model_src_path=odir)
    app.test_on_data(args['checkpoint_file'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_fpath",
                        default='models/config/config_edGNN_graph_class.json')

    parser.add_argument("-g", "--gpu", type=int, default=-1, help="gpu")

    parser.add_argument("action", choices={'train', 'test', 'test_data', 'prep'})

    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size (only for graph classification)")
    parser.add_argument("--k_fold", type=int, default=10,
                        help="k_fold (only for graph classification)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--weight_decay", type=float,
                        default=5e-4, help="Weight for L2 loss")

    # parser.add_argument('test', action='store_true', default=False)
    parser.add_argument("-cp", "--checkpoint_file", default=None)
    # parser.add_argument("-o", "--out_dir", default=None)

    args = vars(parser.parse_args())
    print('args', args)

    # if args.train_tfidf and not args.from_report_folder:
    #     raise AssertionError('Train TF-IDF (-tf) cannot be on when processing from report folder (-fr) is off')

    if args['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        print('cuda', cuda)
        torch.cuda.set_device(args['gpu'])


    config_params = read_params(args['config_fpath'], verbose=False)
    # combine arguments from config file and args
    for key in args:
        config_params[key] = args[key]

    if 'prepend_vocab' in config_params and not config_params['prepend_vocab'] and not config_params['vocab_path']:
        raise AssertionError('prepend_vocab (-pv) cannot be off when no vocab_path is parsed (-v)')


    ###########################
    # Load data
    ###########################
    data_ = load_dataset(config_params, cuda)

    ###########################
    # Run the app
    ###########################
    if args['action'] == "test_data":
        run_app_2(config_params, data_, cuda)
    else:
        run_app(config_params, data_,cuda)