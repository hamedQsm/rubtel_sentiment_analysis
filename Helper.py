import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
# import enchant

import pandas as pd
from sklearn.model_selection import train_test_split

class Helper:
	def __init__(self):
		nltk.download('stopwords')
		nltk.download('wordnet')

		self.stopwords = set(stopwords.words('english'))
		self.punctuation = set(string.punctuation)
		self.lemmatize = WordNetLemmatizer()

		# self.english_dic = enchant.Dict("en_US")

	def preprocess_text(self, text):
		'''
		removes stop words, punctuations, and lemmatizes
		:param text: 
		:return: 
		'''
		stopwords_removed = ' '.join([i for i in str(text).lower().split() if i not in self.stopwords])
		punct_removed = ''.join(i for i in stopwords_removed if i not in self.punctuation)
		lemmatized = " ".join(self.lemmatize.lemmatize(i) for i in punct_removed.split())
		return lemmatized


	def train_test_split(self, df, x_name, y_name, test_size=.3):
		X = df[x_name].values
		y = df[y_name]

		return train_test_split(X, y, test_size=test_size, random_state=0)



    # def is_english(self, text):
    #     '''
    #     Checks if the given text is English. If more than half of the words are English returns true.
    #     :param text: 
    #     :return: 
    #     '''
    #     is_en_arr = [int(self.english_dic.check(w)) for w in str(text).split()]
    #     is_en = sum(is_en_arr) / float(len(is_en_arr)) > .5
    #     return is_en

