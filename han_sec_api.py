import config as cf
# print('cf', cf)
import sys
sys.path.insert(1, cf.__ROOT__)
import os
from utils.inits import to_cuda
from utils.io import print_graph_stats, read_params, create_default_path, remove_model
from utils.constants import *
from han_sec_app import App

# CONFIG_PATH = cf.__ROOT__+'/__save_results/gat_nw__8593__1629__cuckoo_ADung__iapi__vocablower_iapi_new__edge-doc2vec_node-tfidf/config_gat__iapi__edge-doc2vec_node-tfidf_test_data.json'

# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__edgnn__9271__1111__vocablower_iapi_n__tfidf/edgnn_model_test_data.json' # use this
# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__edgnn_w__8958__815__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'

# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__edgnn_w__9219__1778__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'
# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__merge__edgnn_w__8854__963__vocablower_iapi_n__tfidf/edgnn_model_test_data.json'

# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__merge__edgnn_w__9167__1556__vocablower_iapi_m__tfidf__topk=2/edgnn_model_test_data.json'
# from utils.prep_data import PrepareData



# test ----------------
# CONFIG_PATH = cf.__ROOT__+'/__save_results/ft_type+lbl__9266__666__cuckoo_ADung__iapi__vocablower_iapi__doc2vec/config_edGNN_graph_class_test_data.json'

# prep_path = os.path.dirname(CONFIG_PATH)
# print('prep_path', prep_path)
# sys.path.insert(2, prep_path)
# from prep_data import PrepareData


# __REVERSE_EDGE__ = False
# __APPEND_NID_EID__ = False
# __DO_DRAW__ = False
# ----------------


# newest -----------------
# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__merge__edgnn_w__9268__867__vocabnew__tfidf__topk=3/edgnn_n_prep_test_data.json'

CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__TuTu__vocabtutu__iapi__tfidf__topk=10_lv=word/2020-10-12_08-14-43/edgnn.json'
CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__TuTu__vocabtutu__iapi__tfidf__topk=10_lv=word/2020-10-12_15-30-27/edgnn.json'
# CONFIG_PATH = cf.__ROOT__+'/__save_results/reverse__TuTu__vocabtutu__iapi__tfidf__topk=3/9691/config_edGNN_graph_class.json'

prep_path = os.path.dirname(CONFIG_PATH)
print('prep_path', prep_path)
sys.path.insert(2, prep_path)
from prep_data_n import PrepareData

# Overwrite
__REVERSE_EDGE__ = True
__APPEND_NID_EID__ = False
__DO_DRAW__ = False


# CONFIG_PATH = cf.__ROOT__+'/models/configs/config_edGNN_graph_class.json'

# from utils.prep_data_n import PrepareData


class HAN_module:
    def __init__(self, cuckoo_analysis_dir=None, report_dir_name=None, report_dir_path=None, report_file_name=None):
        self.load_args(report_dir_name, report_dir_path, report_file_name)
        self.prep_data = PrepareData(self.args, cuckoo_analysis_dir=cuckoo_analysis_dir)
        self.app = None
    
    def set_task_ids(self, task_ids=None):
        self.task_ids = task_ids
        if task_ids is not None:
            task_ids = [str(tid) for tid in task_ids]
            batch_task_name = '-'.join(task_ids)
            self.args["input_report_folder"] = cf.__ROOT__+'/api_tasks/data_report'
            self.args["input_data_folder"] = cf.__ROOT__+'/api_tasks/data_json/{}'.format(batch_task_name)
            self.args["input_pickle_folder"] = cf.__ROOT__+'/api_tasks/data_pickle/{}'.format(batch_task_name)

            self.prep_data.reset()
            self.prep_data.set_dir(self.args)

        else:
            print('[set_task_ids] task_ids cannot be None !')
    
    def load_args(self, report_dir_name=None, report_dir_path=None, report_file_name=None):
        self.args = read_params(CONFIG_PATH, verbose=False)

        self.report_dir_name = report_dir_name
        self.report_dir_path = report_dir_path
        self.report_file_name = report_file_name

        self.args['config_fpath'] = CONFIG_PATH

        self.args['from_pickle'] = False


        # At initialization, task_ids = None.
        # set folders to these:
        self.args["input_report_folder"] = cf.__ROOT__+'/api_tasks/data_report'
        self.args["input_data_folder"] = cf.__ROOT__+'/api_tasks/data_json/{}'.format(report_dir_name)
        self.args["input_pickle_folder"] = cf.__ROOT__+'/api_tasks/data_pickle/{}'.format(report_dir_name)


        self.args["mapping_path"] = cf.__ROOT__+'/'+self.args["mapping_path"]
        self.args["train_embedding_path"] = cf.__ROOT__+'/'+self.args["train_embedding_path"]
        self.args["vocab_path"] = cf.__ROOT__+'/'+self.args["vocab_path"]

        # self.args["graph_viz_dir"] = cf.__ROOT__+'/data_graphviz'

        # self.args["from_pickle"] = False
        # self.args["from_folder"] = False
        # self.args["from_json"] = False
        
        self.args["prepare_word_embedding"] = True
        self.args["train_embedder"] = False

        self.args["reverse_edge"] = __REVERSE_EDGE__

        del self.args['train_list_file']
        del self.args['test_list_file']

        self.args['do_draw'] = __DO_DRAW__


        # for predict
        fconfig_name = self.args['config_fpath'].split('/')[-1]
        self.odir = self.args['config_fpath'].split(fconfig_name)[0]

        self.model_config = self.args['model_configs']

        self.args['checkpoint_file'] = self.odir+'/checkpoint'

        print('\t [load_args] self.args', self.args)
        print('\t [load_args] self.odir', self.odir)


    def prepare_files(self, cuda=True):
        if self.task_ids is None:
            data = self.prep_data.load_data_files(self.task_ids, report_dir_path=self.report_dir_path, report_dir_name=self.report_dir_name, report_file_name=self.report_file_name)
        else:
            data = self.prep_data.load_data_files(self.task_ids)
        # data = self.prep_data.load_data()

        if data is None:
            return None
        
        data = to_cuda(data) if cuda is True else data
        return data


    def predict_files(self, data, cuda=True):
        # print('\n[han_sec_api][predict_files] *** Start testing ***\n')
        learning_config = {'cuda': cuda}

        graphviz_dir_path = self.args["input_pickle_folder"].replace('data_pickle', 'data_graphviz')
        gdot_path = None if self.args['do_draw'] is False else '{}/{}'.format(graphviz_dir_path, data[GNAMES][0])

        if self.app is None:
            self.app = App(data, model_config=self.model_config,
                            learning_config=learning_config,
                            pretrained_weight=self.args['checkpoint_file'], early_stopping=True, patience=20, 
                            json_path=self.args['input_data_folder'], pickle_folder=self.args['input_pickle_folder'], vocab_path=self.args['vocab_path'],
                            mapping_path=self.args['mapping_path'],
                            model_src_path=self.odir,
                            append_nid_eid=__APPEND_NID_EID__,
                            gdot_path=gdot_path
                        )
            self.app.load_model_state(model_path=self.args['checkpoint_file'])
        else:
            self.load_data(data)
        return self.app.predict()



if __name__ == "__main__":
    # cuda = True

    # data, self.args = prepare_files([16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    # data, self.args = prepare_files([99,100,101,102,103,   22,23,24,25,26,27,28,29,   50,51,52,53,54,55,56])
    # data, self.args = prepare_files([31]) # should output [1,0]
    # data, self.args = prepare_files([50,51,52,53,54,55,56]) # all 1
    # data, self.args = prepare_files([103])
    # data, self.args = prepare_files([99,100,101,102,103]) # all 0

    # data, self.args = prepare_files([112, 118, 123]) # 0 ok
    # data, self.args = prepare_files([132, 133, 136, 140]) # 1 ok
    # data, self.args = prepare_files([112, 118, 123,  132, 133, 136, 140]) 

    # data, self.args = prepare_files([161,162,163]) # doc

    # cuda = False

    # data, self.args = prepare_files([4], cuda) # VirusShare_0d93334c773fb884da7b536d89de2962__PE
    # data, self.args = prepare_files([5], cuda) # VirusShare_0cfd0c18dfea446bf2ebd7e80d047b8a__nullsoft_0.65
    # data, self.args = prepare_files([6], cuda) # VirusShare_0a7684ed716f2bc360990115b781cc8f__trojan__PE__0.58
    # data, self.args = prepare_files([7], cuda) # Microsoft.Build.Conversion.v4.0.dll
    # data, self.args = prepare_files([8], cuda) # Microsoft.Build.Tasks.v4.0.dll

    # if data is None:
    #     print('Graph can\'t be created!')
    # else:
    #     labels, scores = predict_files(data, self.args, cuda)
    #     print(labels, scores)



    cuda = True

    # benigns
    tasks = [1746, 1748, 1750, 1751, 1754, 1756]
    # malware
    tasks = [247, 303, 304, 310, 312, 1655, 1656, 1657, 1659, 1660]


    tasks = [1746, 1748, 1750, 1751, 1754, 1756,
            247, 303, 304, 310, 312, 1655, 1656, 1657, 1659, 1660]

    # tasks = [9819]

    tasks = None
    # han = HAN_module(task_ids=tasks, report_dir_name='game_Linh', report_file_name=None)


    rp_folder = '/media/tunguyen/TuTu_Passport/MTAAV/HAN-sec-new/api_tasks/data_report'
    rp_dir_name = '2870_benign'
    rp_dir_name = 'game_Linh'
    rp_dir_name = '2863_benign'
    rp_dir_name = '9819_malware__new'
    # rp_dir_name = 'mal'

    # If task_ids != None:
    #    report_dir_name, report_file_name, report_dir_path = None
    #    cuckoo_analysis_dir != None
    # han = HAN_module(cuckoo_analysis_dir=rp_folder, 
    #                  report_dir_name=rp_dir_name, 
    #                  report_file_name=None
    #                 #  cuckoo_analysis_dir='/home/mtaav/.cuckoo/storage/analyses', 
    #                 #  report_dir_name='hh', 
    #                 )

    # '''
    # # tasks = [9249, 9254, 9255] #
    # # tasks = [9256, 9257, 9258, 9259] # 0 1 1 1
    # # tasks = [9260, 9261, 9262]
    # tasks = [9305]
    # han = HAN_module(task_ids=tasks)

    # cuda = False
    # '''

    # data = han.prepare_files(cuda=cuda) # Microsoft.Build.Tasks.v4.0.dll
    # if data is None:
    #     print('Graph can\'t be created!')
    # else:
    #     labels, scores = han.predict_files(data, cuda=cuda)
    #     print('labels', labels)
    #     print('scores', scores)





    han = HAN_module(cuckoo_analysis_dir=rp_folder)
    task_ids_list = [ [10204],
                      [10205]
                    #   [2861, 2863, 2870],
                    #   [2861, 2863],
                    #   [2863, 2870]
                    ]
    for task_ids in task_ids_list:
        han.set_task_ids(task_ids=task_ids)
        data = han.prepare_files(cuda=cuda) # Microsoft.Build.Tasks.v4.0.dll
        if data is None:
            print('Graph can\'t be created!')
        else:
            print('-------- Call predict_files for', task_ids, '--------')
            labels, scores = han.predict_files(data, cuda=cuda)
            print('labels', labels)
            print('scores', scores)
