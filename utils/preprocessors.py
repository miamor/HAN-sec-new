import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

import utils.utils

class CSVPreprocessor:
    
    def __init__(self, dataset_dir, split_train_test=True):
        self.split_train_test = split_train_test
        
        if split_train_test:
            self.train_csv = os.path.join(dataset_dir, 'train.csv')
            self.val_csv = os.path.join(dataset_dir, 'val.csv')
            self.test_csv = os.path.join(dataset_dir, 'test.csv')
            self.classes_txt = os.path.join(dataset_dir, 'classes.txt')
        else:
            self.corpus_csv = os.path.join(dataset_dir, 'corpus.csv')
            self.classes_txt = os.path.join(dataset_dir, 'classes.txt')

        self.columns = ['class', 'data', 'file']
        
    def preprocess(self, preprocess_level='word', val_size=0.1):
        # print('[preprocessors][preprocess] self.split_train_test', self.split_train_test)
       
        assert preprocess_level in ['word', 'char'], "preprocess_level should be either 'word' or 'char'"
        
        if self.split_train_test:
            # return self.preprocess_train_test_val(preprocess_level)
            return self.preprocess_train_test(preprocess_level)
        else:
            return self.preprocess_all(preprocess_level)


    def preprocess_train_test_val(self, preprocess_level='word'):
                
        train_df = (pd.read_csv(self.train_csv, names=self.columns)
                     .assign(label=lambda x: x['class'].astype(int)))
        train_data = self._dataframe_to_data(train_df, preprocess_level)

        val_df = (pd.read_csv(self.val_csv, names=self.columns)
                     .assign(label=lambda x: x['class'].astype(int)))
        val_data = self._dataframe_to_data(val_df, preprocess_level)
        
        test_df = (pd.read_csv(self.test_csv, names=self.columns)
                     .assign(label=lambda x: x['class'].astype(int)))
        test_data = self._dataframe_to_data(test_df, preprocess_level)
            
        return train_data, val_data, test_data

    def preprocess_train_test(self, preprocess_level='word'):
                
        train_df = (pd.read_csv(self.train_csv, names=self.columns)
                     .assign(label=lambda x: x['class'].astype(int))
                    )
        
        train_data = self._dataframe_to_data(train_df, preprocess_level)
        
        test_df = (pd.read_csv(self.test_csv, names=self.columns)
                     .assign(label=lambda x: x['class'].astype(int))
                    )
        test_data = self._dataframe_to_data(test_df, preprocess_level)
            
        return train_data, test_data

    def preprocess_all(self, preprocess_level='word'):
        # print('[preprocessors][preprocess_all] self.corpus_csv', self.corpus_csv)
                
        _df = (pd.read_csv(self.corpus_csv, names=self.columns)
                     .assign(label=lambda x: x['class'].astype(int))
                    )
        
        _data = self._dataframe_to_data(_df, preprocess_level)
            
        return _data


    @staticmethod
    def _dataframe_to_data(dataframe, preprocess_level):
        dataframe = dataframe.dropna(subset=['data', 'label'])
        if preprocess_level == 'word':
            dataframe = dataframe.assign(text=lambda df: df['data'].map(lambda text: text.split()))
        elif preprocess_level == 'char':
            pass
        # print('\ntitle', dataframe['title'])
        # print('text', dataframe['text'])
        # data = [(text, label) for text, label in zip(dataframe['title']+' '+dataframe['text'], dataframe['label'])]
        data = [(text, label) for text, label in zip(dataframe['data'], dataframe['label'])]
        return data
