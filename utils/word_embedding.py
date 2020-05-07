from tqdm import tqdm
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re
from sklearn.model_selection import train_test_split


import numpy as np
import csv
from utils.preprocessors import CSVPreprocessor

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
import pickle

import json


class WordEmbedding(object):
    
    embedding_data_csv = {}
    mapping = {}
    embedding_data_csv__cls = {}
    flags_keys = []

    train_size = 0.8
    val_size = 0.1
    # test_size = 0.1

    word_to_ix = {}

    tfidf = None
    max_ft = 100000
    top_k = 3

    def __init__(self, emb_trained_path, emb_corpus_path, mapping):
        self.emb_trained_path = emb_trained_path
        self.emb_corpus_path = emb_corpus_path
        self.mapping = mapping
        self.type = emb_corpus_path.split('/')[-1].split('_')[0]

        return

    def save_corpus(self, embedding_data_csv):
        ''' Save corpus to pickle folder '''
        with open('{}/corpus.csv'.format(self.emb_corpus_path), 'w') as f:
            # fnames = ['class', 'data']
            fnames = ['class', 'data', 'file']
            writer = csv.DictWriter(f, fieldnames=fnames)

            for index, data in enumerate(embedding_data_csv):
                writer.writerow(data)

        with open('{}/classes.txt'.format(self.emb_corpus_path), 'w') as f:
            f.write('\n'.join(self.mapping.keys()))
        with open('{}/flags_keys.txt'.format(self.emb_corpus_path), 'w') as f:
            f.write('\n'.join(self.flags_keys))


    def prepare_(self, embedding_data_csv, embedding_data_csv__cls, flags_keys=None):
        ''' Save data for training tf-idf '''
        # split train/test of each class
        emb_data_by_type = {
            'train': [],
            'val': [],
            'test': []
        }
        for cls_name in self.mapping.keys():
            emb_data = embedding_data_csv__cls[cls_name]
            N = len(emb_data)
            end_idx_train = int(self.train_size*N)
            end_idx_val = int(self.val_size*N) + end_idx_train
            # print(cls_name, 'end_idx_train', end_idx_train)
            # print(cls_name, 'end_idx_val', end_idx_train)
            for index, data in enumerate(emb_data):
                if index < end_idx_train:
                    emb_data_by_type['train'].append(data)
                elif index < end_idx_val:
                    emb_data_by_type['val'].append(data)
                else:
                    emb_data_by_type['test'].append(data)

        for dtype in emb_data_by_type:
            with open('{}/{}.csv'.format(self.emb_trained_path, dtype), 'w') as f:
                fnames = ['class', 'data']
                writer = csv.DictWriter(f, fieldnames=fnames)

                for index, data in enumerate(emb_data_by_type[dtype]):
                    writer.writerow(data)

        # Save corpus to embedding trained path
        with open('{}/corpus.csv'.format(self.emb_trained_path), 'w') as f:
            fnames = ['class', 'data']
            writer = csv.DictWriter(f, fieldnames=fnames)

            for index, data in enumerate(embedding_data_csv):
                writer.writerow(data)


        with open('{}/classes.txt'.format(self.emb_trained_path), 'w') as f:
            f.write('\n'.join(self.mapping.keys()))
        
        if flags_keys is not None:
            with open('{}/flags_keys.txt'.format(self.emb_trained_path), 'w') as f:
                f.write('\n'.join(self.flags_keys))


    def prepare(self, embedding_data_csv, train_list_name, test_list_name, flags_keys=None):
        ''' Save data for training tf-idf '''
        # split train/test of each class
        emb_data_by_type = {
            'train': [],
            'test': []
        }
        # for cls_name in self.mapping.keys():
        #     emb_data = embedding_data_csv__cls[cls_name]
        #     for index, data in enumerate(emb_data):
        #         print('data in emb_data', data)
        #         if data['file'] in train_list_name:
        #             emb_data_by_type['train'].append(data)
        #         elif data['file'] in test_list_name:
        #             emb_data_by_type['test'].append(data)
        for index, data in enumerate(embedding_data_csv):
            # print('data in emb_data', data)
            if data['file'] in train_list_name:
                emb_data_by_type['train'].append(data)
            elif data['file'] in test_list_name:
                emb_data_by_type['test'].append(data)

        for dtype in emb_data_by_type:
            with open('{}/{}.csv'.format(self.emb_trained_path, dtype), 'w') as f:
                fnames = ['class', 'data', 'file']
                writer = csv.DictWriter(f, fieldnames=fnames)

                for index, data in enumerate(emb_data_by_type[dtype]):
                    writer.writerow(data)

        # Save corpus to embedding trained path
        with open('{}/corpus.csv'.format(self.emb_trained_path), 'w') as f:
            fnames = ['class', 'data', 'file']
            writer = csv.DictWriter(f, fieldnames=fnames)

            for index, data in enumerate(embedding_data_csv):
                writer.writerow(data)


        with open('{}/classes.txt'.format(self.emb_trained_path), 'w') as f:
            f.write('\n'.join(self.mapping.keys()))
        
        if flags_keys is not None:
            with open('{}/flags_keys.txt'.format(self.emb_trained_path), 'w') as f:
                f.write('\n'.join(self.flags_keys))



    # def prepare_corpus(self, embedding_data_csv, flags_keys=None):
    #     ''' Save data for loading vectorizer '''
    #     # Save corpus to embedding corpus path
    #     with open('{}/corpus.csv'.format(self.emb_corpus_path), 'w') as f:
    #         fnames = ['class', 'data', 'file']
    #         writer = csv.DictWriter(f, fieldnames=fnames)

    #         for index, data in enumerate(embedding_data_csv):
    #             writer.writerow(data)


    #     with open('{}/classes.txt'.format(self.emb_corpus_path), 'w') as f:
    #         f.write('\n'.join(self.mapping.keys()))
        
    #     if flags_keys is not None:
    #         with open('{}/flags_keys.txt'.format(self.emb_corpus_path), 'w') as f:
    #             f.write('\n'.join(self.flags_keys))



class Doc2Vec_(WordEmbedding):

    num_epochs = 30
    model_dbow = None
    vector_size = None
    dm = None

    def __init__(self, emb_trained_path, emb_corpus_path, mapping, vector_size, dm):
        super(Doc2Vec_, self).__init__(emb_trained_path, emb_corpus_path, mapping)
        if vector_size is None or vector_size <= 0:
            raise AssertionError("vector_size must be set and > 0")
        if dm is None or dm not in [0, 1]:
            raise AssertionError("dm must be set to 0 or 1")
        self.vector_size = vector_size
        self.dm = dm


    def label_sentences(self, corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(TaggedDocument(v.split(' '), [label]))
        return labeled



    def train(self):
        ''' Train doc2vec '''
        print('\nTrain doc2vec for', self.emb_trained_path)
        preprocessor = CSVPreprocessor(self.emb_trained_path)
        # train_data, val_data, test_data = preprocessor.preprocess(preprocess_level='char')
        # train_val_data = train_data + val_data
        train_val_data, test_data = preprocessor.preprocess(preprocess_level='char')
        # print('train_data', len(train_data))
        # print('val_data', len(val_data))
        print('train_val_data', len(train_val_data))
        print('test_data', len(test_data))

        X_train = [text for text, label in train_val_data]
        X_test = [text for text, label in test_data]
        y_train = [label for text, label in train_val_data]
        y_test = [label for text, label in test_data]

        X_train = self.label_sentences(X_train, 'Train')
        X_test = self.label_sentences(X_test, 'Test')
        all_data = X_train + X_test

        '''
        Parameters
            dm=0 , distributed bag of words (DBOW) is used.
            vector_size=300 , 300 vector dimensional feature vectors.
            negative=5 , specifies how many “noise words” should be drawn.
            min_count=1, ignores all words with total frequency lower than this.
            alpha=0.065 , the initial learning rate.
        '''
        self.model_dbow = Doc2Vec(dm=self.dm, vector_size=self.vector_size, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
        self.model_dbow.build_vocab([x for x in tqdm(all_data, disable=True)])

        for epoch in range(self.num_epochs):
            self.model_dbow.train(utils.shuffle([x for x in tqdm(all_data, disable=True)]), total_examples=len(all_data), epochs=1)
            self.model_dbow.alpha -= 0.002
            self.model_dbow.min_alpha = self.model_dbow.alpha


        train_vectors_dbow = self.get_vectors(self.model_dbow, len(X_train), self.vector_size, 'Train')

        # fit the model
        model = LogisticRegressionCV(max_iter=10000)
        model.fit(train_vectors_dbow, y_train) # train model

        # Infer
        self.test_on_set('Train~', X_train, y_train, self.model_dbow, model, 'Train')
        self.test_on_set('Test~', X_test, y_test, self.model_dbow, model, 'Test')

        # Save doc2vec features and model
        print('Save to {}/{}__d2v_vectorize.pkl'.format(self.emb_trained_path, self.type))
        self.model_dbow.save("{}/{}__d2v_vectorize.pkl".format(self.emb_trained_path, self.type))
        with open("{}/{}__d2v_model.pkl".format(self.emb_trained_path, self.type), 'wb') as handle:
            pickle.dump(model, handle)
        
        return self.get_dict_vec()


    def get_vectors(self, model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors


    def test_on_set(self, set_name, X, y, vectorizer, model, prefix=None):
        if prefix is not None:
            X_vectors = self.get_vectors(vectorizer, len(X), self.vector_size, prefix)
        else:
            X_vectors = [vectorizer.infer_vector(x.split(' ')) for x in X]

        score = model.score(X_vectors, y)

        result_base = "Doc2Vec Accuracy on set {}: {acc:<.1%}"
        result = result_base.format(set_name, acc=score)
        print(result)

        return X_vectors


    def load(self, load_train_test_set=True):
        print('\nLoad doc2vec model for', self.emb_corpus_path)

        self.model_dbow = Doc2Vec.load("{}/{}__d2v_vectorize.pkl".format(self.emb_trained_path, self.type))
        model = pickle.load(open("{}/{}__d2v_model.pkl".format(self.emb_trained_path, self.type), 'rb'))


        ''' Load and test on this corpus set '''
        preprocessor = CSVPreprocessor(self.emb_corpus_path, False)
        corpus = preprocessor.preprocess(preprocess_level='char')
        
        X = [text for text, label in corpus]
        y = [label for text, label in corpus]

        # print('Corpus:', self.emb_corpus_path)
        # self.test_on_set('Corpus~', X, y, self.model_dbow, model)


        ''' Load and test on train/test set '''
        if load_train_test_set:
            preprocessor = CSVPreprocessor(self.emb_trained_path)
            # train_data, val_data, test_data = preprocessor.preprocess(preprocess_level='char')
            # train_val_data = train_data + val_data
            train_val_data, test_data = preprocessor.preprocess(preprocess_level='char')
            
            X_train = [text for text, label in train_val_data]
            X_test = [text for text, label in test_data]
            y_train = [label for text, label in train_val_data]
            y_test = [label for text, label in test_data]

            X_train = self.label_sentences(X_train, 'Train')
            X_test = self.label_sentences(X_test, 'Test')

            self.test_on_set('Train~', X_train, y_train, self.model_dbow, model, 'Train')
            self.test_on_set('Test~', X_test, y_test, self.model_dbow, model, 'Test')

            return self.get_dict_vec()

        return None, None


    def get_dict_vec(self):
        for word in self.model_dbow.wv.vocab:
            self.word_to_ix[word] = self.model_dbow.wv.word_vec(word)
        # print('self.word_to_ix', self.word_to_ix)

        # Save (one word is a vector ~ cannot save yet)
        # with open(self.emb_corpus_path+'/word_to_ix.json', 'w') as f:
        #     json.dump(self.word_to_ix, f)

        return self.word_to_ix, self.model_dbow.wv.vocab


    def transform(self, text):
        text = text.lower().split(' ')
        dbow_vectors = self.model_dbow.infer_vector(text)
        return dbow_vectors


class TFIDF(WordEmbedding):
    
    tfidf = None
    max_ft = None #100000
    top_k = None
    ngrams = (1, 1) #(1, 5)

    def __init__(self, emb_trained_path, emb_corpus_path, mapping, max_ft, top_k):
        super(TFIDF, self).__init__(emb_trained_path, emb_corpus_path, mapping)
        print('max_ft', max_ft)
        if max_ft is None or max_ft <= 0:
            raise AssertionError("max_ft must be set and > 0")
        if top_k is None or top_k <= 0:
            raise AssertionError("top_k must be set and > 0")
        self.max_ft = max_ft
        self.top_k = top_k


    def test_on_set(self, set_name, X, y, vectorizer, model):
        x_transformed = vectorizer.transform(X)
        
        score = model.score(x_transformed, y)

        result_base = "TF-IDF Accuracy on set {}: {acc:<.1%}"
        result = result_base.format(set_name, acc=score)
        print(result)

        return x_transformed


    def train(self):
        ''' Train tf-idf '''
        print('\nTrain TF-IDF for', self.emb_trained_path)
        preprocessor = CSVPreprocessor(self.emb_trained_path)
        # train_data, val_data, test_data = preprocessor.preprocess(preprocess_level='char')
        # train_val_data = train_data + val_data
        train_val_data, test_data = preprocessor.preprocess(preprocess_level='char')
        # print('train_data', len(train_val_data))
        # print('val_data', len(val_data))
        print('train_val_data', len(train_val_data))
        print('test_data', len(test_data))

        ''' Load and test on train/test set '''
        x_train = [text for text, label in train_val_data]
        y_train = [label for text, label in train_val_data]
        x_test = [text for text, label in test_data]
        y_test = [label for text, label in test_data]


        self.tfidf = TfidfVectorizer(ngram_range=self.ngrams)
        x_train_transformed = self.tfidf.fit_transform(x_train)
        # self.tfidf.fit(x_train)

        ''' Train a simple classifier '''
        model = LogisticRegressionCV(max_iter=10000)
        model.fit(x_train_transformed, y_train) # train model

        x_transformed = self.test_on_set('Train~', x_train, y_train, self.tfidf, model)
        self.test_on_set('Test~', x_test, y_test, self.tfidf, model)

        # print('self.tfidf.vocabulary_', self.tfidf.vocabulary_)

        # Save tfidf features and model
        print('Save to {}/{}__tfidf_vectorize.pkl'.format(self.emb_trained_path, self.type))
        with open("{}/{}__tfidf_vectorize.pkl".format(self.emb_trained_path, self.type), 'wb') as handle:
            pickle.dump(self.tfidf, handle)
        with open("{}/{}__tfidf_model.pkl".format(self.emb_trained_path, self.type), 'wb') as handle:
            pickle.dump(model, handle)

        return self.get_dict_vec(x_transformed)


    def load(self, load_train_test_set=True):
        print('\nLoad TF-IDF model for', self.emb_corpus_path)

        # Load vectorize and model
        tf_saved = pickle.load(open("{}/{}__tfidf_vectorize.pkl".format(self.emb_trained_path, self.type), 'rb'))
        model = pickle.load(open("{}/{}__tfidf_model.pkl".format(self.emb_trained_path, self.type), 'rb'))


        # Create new tfidfVectorizer with old vocabulary
        self.tfidf = TfidfVectorizer(ngram_range=self.ngrams, vocabulary=tf_saved.vocabulary_)


        ''' Load and test on this corpus set '''
        preprocessor = CSVPreprocessor(self.emb_corpus_path, False)
        corpus = preprocessor.preprocess(preprocess_level='char')
        
        X = [text for text, label in corpus]
        y = [label for text, label in corpus]


        # if load_train_test_set:
        if True: # tfidf always requires re-fit 
            preprocessor = CSVPreprocessor(self.emb_trained_path)
            # train_data, val_data, test_data = preprocessor.preprocess(preprocess_level='char')
            # train_val_data = train_data + val_data
            train_val_data, test_data = preprocessor.preprocess(preprocess_level='char')

            x_train = [text for text, label in train_val_data]
            y_train = [label for text, label in train_val_data]
            x_test = [text for text, label in test_data]
            y_test = [label for text, label in test_data]

            # print('x_train', x_train)
            self.tfidf.fit(x_train)
            x_transformed = self.test_on_set('Train~', x_train, y_train, self.tfidf, model)
            self.test_on_set('Test~', x_test, y_test, self.tfidf, model)

            # self.test_on_set('Corpus~', X, y, self.tfidf, model)

            return self.get_dict_vec(x_transformed)

        return None, None


    def get_dict_vec(self, x_transformed):
        '''
        Since tf-idf value is for each document, we use this to calculate the average tf-idf in whole corpus
        '''
        # print('x_transformed', x_transformed)
        tfidf_vectors = x_transformed.todense()
        # tfidf_vectors of words not in the doc will be 0, so replace them with nan
        tfidf_vectors[tfidf_vectors == 0] = np.nan
        # Use nanmean of numpy which will ignore nan while calculating the mean
        tfidf_means = np.nanmean(tfidf_vectors, axis=0)
        # convert it into a dictionary for later lookup
        tfidf_means = dict(zip(self.tfidf.get_feature_names(), tfidf_means.tolist()[0]))
        # print('tfidf_means', tfidf_means)

        self.words = self.tfidf.get_feature_names()

        # create dictionary to find a tfidf word each word
        for word in self.words:
            self.word_to_ix[word] = tfidf_means[word]

        # print('self.word_to_ix', self.word_to_ix)

        # Save
        # print('self.word_to_ix', self.word_to_ix)
        # with open(self.emb_corpus_path+'/word_to_ix.json', 'w') as f:
        #     json.dump(self.word_to_ix, f)


        tfidf_vectors_to_sort = x_transformed.todense()
        self.tfidf_ordered = np.argsort(tfidf_vectors_to_sort*-1)


        self.words_unique = []
        # print('self.word_to_ix', self.word_to_ix, 'self.word_to_ix')
        # print('self.tfidf_ordered', self.tfidf_ordered, 'self.tfidf_ordered')
        for w in self.words:
            if w not in self.words_unique:
                self.words_unique.append(w)

        return self.word_to_ix, self.words_unique


    # def transform(self, text):
    #     top_k = 5
    #     for i, doc in enumerate(docs):
    #         result = { }
    #         # Pick top_k from each argsorted matrix for each doc
    #         for t in range(top_k):
    #             # Pick the top k word, find its average tfidf from the
    #             # precomputed dictionary using nanmean and save it to later use
    #             result[self.words[self.tfidf_ordered[i,t]]] = means[self.words[self.tfidf_ordered[i,t]]]
    #         print(result)

    def transform(self, text):
        x = text.split(' ')

        if len(x) < self.top_k:
            for i in range(self.top_k-len(x)):
                x.append('')
        
        # get self.top_k (eg: 3) top max tf-idf scores
        # print('self.word_to_ix', self.word_to_ix)
        v_all = []
        txt_chosens = []
        for txt in x:
            if txt not in self.word_to_ix:
                # print('{} not in self.word_to_ix'.format(txt))
                v_all.append(0.0)
            else:
                v_all.append(self.word_to_ix[txt])

        # Select top_k max 
        v_all = np.array(v_all)
        max_indices = v_all.argsort()[-self.top_k:][::-1]
        vmax = v_all[max_indices]
        for idx in max_indices:
            txt_chosens.append(x[idx])

        # Select top_k - 1 min (remove the smallest)
        # min_indices = v_all.argsort()[1:self.top_k][::-1]
        # vmin = v_all[min_indices]
        # for idx in min_indices:
        #     txt_chosens.append(x[idx])
        
        # Concat to get top_k*2-1 dim vector
        # vget = np.concatenate((vmax, vmin))
        vget = vmax
        # print('vget', vget)


        # print('\nx', x)
        # print('max_indices', max_indices)
        # print('vmax', vmax)
        # print('txt_chosens', txt_chosens)

        # return vmax, txt_chosens
        return vget

        # x_transformed = self.tfidf.transform(x)
        # x_transformed = x_transformed.todense()
        # return x_transformed