__ROOT__ = '/media/fitmta/Storage/MinhTu/HAN_sec_new'
import sys
sys.path.insert(0, __ROOT__)
from utils.prep_data import PrepareData
from utils.inits import to_cuda
from utils.io import print_graph_stats, read_params, create_default_path, remove_model
from utils.constants import *
from han_sec_app import App

# CONFIG_PATH = __ROOT__+'/__save_results/gat_nw__8593__1629__cuckoo_ADung__iapi__vocablower_iapi_new__edge-doc2vec_node-tfidf/config_gat__iapi__edge-doc2vec_node-tfidf_test_data.json'

# CONFIG_PATH = __ROOT__+'/__save_results/reverse__edgnn__9271__1111__vocablower_iapi_n__tfidf/edgnn_model_test_data.json' # use this
CONFIG_PATH = __ROOT__+'/__save_results/reverse__edgnn_w__8958__815__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'

# CONFIG_PATH = __ROOT__+'/__save_results/reverse__edgnn_w__9219__1778__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'
# CONFIG_PATH = __ROOT__+'/__save_results/reverse__merge__edgnn_w__8854__963__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'

# CONFIG_PATH = __ROOT__+'/__save_results/reverse__merge__edgnn_w__9167__1556__vocablower_iapi_m__tfidf__topk=2/edgnn_model_test_data.json'
__REVERSE_EDGE__ = True
__APPEND_NID_EID__ = True
__DO_DRAW__ = True

def prepare_files(task_ids=None, cuda=True):
    args = read_params(CONFIG_PATH, verbose=False)

    args['config_fpath'] = CONFIG_PATH

    if task_ids is not None:
        task_ids = [str(tid) for tid in task_ids]
        batch_task_name = '-'.join(task_ids)
        args["input_report_folder"] = __ROOT__+'/api_tasks/data_report'
        args["input_data_folder"] = __ROOT__+'/api_tasks/data_json/{}'.format(batch_task_name)
        args["input_pickle_folder"] = __ROOT__+'/api_tasks/data_pickle/{}'.format(batch_task_name)
    
    args["mapping_path"] = __ROOT__+'/'+args["mapping_path"]
    args["train_embedding_path"] = __ROOT__+'/'+args["train_embedding_path"]
    args["vocab_path"] = __ROOT__+'/'+args["vocab_path"]

    args["graph_viz_dir"] = __ROOT__+'/data/graphviz'

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
    
    data = to_cuda(data) if cuda else data
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
    # data, args = prepare_files([16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    # data, args = prepare_files([99,100,101,102,103,   22,23,24,25,26,27,28,29,   50,51,52,53,54,55,56])
    # data, args = prepare_files([31]) # should output [1,0]
    # data, args = prepare_files([50,51,52,53,54,55,56]) # all 1
    # data, args = prepare_files([103])
    # data, args = prepare_files([99,100,101,102,103]) # all 0

    # data, args = prepare_files([112, 118, 123]) # 0 ok
    # data, args = prepare_files([132, 133, 136, 140]) # 1 ok
    # data, args = prepare_files([112, 118, 123,  132, 133, 136, 140]) 

    data, args = prepare_files([161,162,163]) # doc

    data, args = prepare_files([-2])

    if data is None:
        print('Graph can\'t be created!')
    else:
        labels, scores = predict_files(data, args)
        print(labels, scores)