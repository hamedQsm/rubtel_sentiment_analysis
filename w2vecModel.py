from collections import defaultdict

import numpy as np
import gensim


# TODO: try tf-idf as well
from sklearn.feature_extraction.text import TfidfVectorizer


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec[next(iter(word2vec))])
        print('DIM: ', self.dim)

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec[next(iter(word2vec))])
        print('DIM: ', self.dim)

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        print (type(self.word2vec))
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


class Word2vecModel:
    def initialize(self, embedding_path):
        '''
        intitialize/loads the w2v model
        :param embedding_path: path to the already trained GloVe model
        :return: 
        '''
        self.w2v = {}
        with open(embedding_path, "rb") as lines:
            for line in lines:
                word = str(line.split()[0])
                # removing b' and ' from the beginning and ending of the word
                word = word[2: -1]
                vec = np.array(list(map(float, line.split()[1:])))
                self.w2v[word] = vec
            
    def train(self, X, dim):
        self.model = gensim.models.Word2Vec(X, size=dim)
        self.w2v = dict(zip(self.model.wv.index2word, self.model.wv.syn0))  

    def transform(self, doc_list, embedding='mean'):
        '''
        transforms a given document list to w2v space
        :param embedding: mean or tfidf (how to assign weights to words for computing embedding)
        :param doc_list: is a list of list, i.e. list of documents.
        :return: 
        '''
        if embedding == 'mean':
            print('Obtaining the mean embeddings...')
            X_embeded = MeanEmbeddingVectorizer(self.w2v).transform(doc_list)
        elif embedding == 'tfidf':
            print('Obtaining the tfidf embeddings...')
            X_embeded = TfidfEmbeddingVectorizer(self.w2v).fit(doc_list).transform(doc_list)
        else:
            X_embeded = None

        return X_embeded
