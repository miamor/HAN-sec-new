import config as cf
import sys
sys.path.insert(0, cf.__ROOT__)
import os
from utils.inits import to_cuda
from utils.io import print_graph_stats, read_params, create_default_path, remove_model
from utils.constants import *
from han_sec_app import App

# CONFIG_PATH = cf.__ROOT__+'/__save_results/gat_nw__8593__1629__cuckoo_ADung__iapi__vocablower_iapi_new__edge-doc2vec_node-tfidf/config_gat__iapi__edge-doc2vec_node-tfidf_test_data.json'

CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__edgnn__9271__1111__vocablower_iapi_n__tfidf/edgnn_model_test_data.json' # use this
# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__edgnn_w__8958__815__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'

# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__edgnn_w__9219__1778__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'
# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__merge__edgnn_w__8854__963__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'

# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__merge__edgnn_w__9167__1556__vocablower_iapi_m__tfidf__topk=2/edgnn_model_test_data.json'
# from utils.prep_data import PrepareData


# newest -----------------
CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__merge__edgnn_w__9268__867__vocabnew__tfidf__topk=3/edgnn_n_prep_test_data.json'
prep_path = os.path.dirname(CONFIG_PATH)
print('prep_path', prep_path)
sys.path.insert(0, prep_path)
from prep_data_n import PrepareData

__REVERSE_EDGE__ = True
__APPEND_NID_EID__ = True
__DO_DRAW__ = False

def prepare_files(task_ids=None, cuda=True):
    args = read_params(CONFIG_PATH, verbose=False)

    args['config_fpath'] = CONFIG_PATH

    if task_ids is not None:
        task_ids = [str(tid) for tid in task_ids]
        batch_task_name = '-'.join(task_ids)
        args["input_report_folder"] = cf.__ROOT__+'/api_tasks/data_report'
        args["input_data_folder"] = cf.__ROOT__+'/api_tasks/data_json/{}'.format(batch_task_name)
        args["input_pickle_folder"] = cf.__ROOT__+'/api_tasks/data_pickle/{}'.format(batch_task_name)
    
    args["mapping_path"] = cf.__ROOT__+'/'+args["mapping_path"]
    args["train_embedding_path"] = cf.__ROOT__+'/'+args["train_embedding_path"]
    args["vocab_path"] = cf.__ROOT__+'/'+args["vocab_path"]

    args["graph_viz_dir"] = cf.__ROOT__+'/data/graphviz'

    args["from_pickle"] = False
    args["from_report_folder"] = False
    args["from_data_json"] = False
    
    args["prepare_word_embedding"] = True
    args["train_embedder"] = False

    args["reverse_edge"] = __REVERSE_EDGE__

    del args['train_list_file']
    del args['test_list_file']

    args['do_draw'] = __DO_DRAW__

    prep_data = PrepareData(args)
    data = prep_data.load_data_files(task_ids)
    if data is None:
        return None, args
    
    data = to_cuda(data) if cuda is True else data
    return data, args


def predict_files(data, args, cuda=True):
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
              model_src_path=odir,
              append_nid_eid=__APPEND_NID_EID__,
            #   gdot_path='{}/data_report/{}'.format(args["graph_viz_dir"], data[GNAMES][0])
            )
    return app.predict(args['checkpoint_file'])



if __name__ == "__main__":
    cuda = True

    # data, args = prepare_files([16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    # data, args = prepare_files([99,100,101,102,103,   22,23,24,25,26,27,28,29,   50,51,52,53,54,55,56])
    # data, args = prepare_files([31]) # should output [1,0]
    # data, args = prepare_files([50,51,52,53,54,55,56]) # all 1
    # data, args = prepare_files([103])
    # data, args = prepare_files([99,100,101,102,103]) # all 0

    # data, args = prepare_files([112, 118, 123]) # 0 ok
    # data, args = prepare_files([132, 133, 136, 140]) # 1 ok
    # data, args = prepare_files([112, 118, 123,  132, 133, 136, 140]) 

    # data, args = prepare_files([161,162,163]) # doc

    cuda = False

    data, args = prepare_files([4], cuda) # VirusShare_0d93334c773fb884da7b536d89de2962__PE
    data, args = prepare_files([5], cuda) # VirusShare_0cfd0c18dfea446bf2ebd7e80d047b8a__nullsoft_0.65
    data, args = prepare_files([6], cuda) # VirusShare_0a7684ed716f2bc360990115b781cc8f__trojan__PE__0.58
    data, args = prepare_files([7], cuda) # Microsoft.Build.Conversion.v4.0.dll
    data, args = prepare_files([8], cuda) # Microsoft.Build.Tasks.v4.0.dll

    if data is None:
        print('Graph can\'t be created!')
    else:
        labels, scores = predict_files(data, args, cuda)
        print(labels, scores)