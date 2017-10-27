import os
import json
from csv import DictReader, DictWriter

import numpy as np
from numpy import array
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer

SEED = 5


'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1
        return features

class year_mentioned(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.pattern4= re.compile("[0-9]+")
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        count=0
        for ex in examples:
            count = len(re.findall(self.pattern4,ex))
            features[i,0] = count
            i += 1

        return features

class negative_sense(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.neg_words = ['neither','nor','not','never','no','none',"don't","isn't","can't","hadn't"]
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        count=0
        for ex in examples:
            sentences = ex.split(".")
            for itr in range(len(sentences)):
                for nve in self.neg_words:
                    if nve in sentences[itr]:
                        count+=1
            features[i,0] = count
            i += 1

        return features


class starts_with_questioning(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.pattern1 = re.compile("(how.*?[?])")
        self.pattern2 = re.compile("(what.*?[?])")
        self.pattern3 = re.compile("(why.*?[?])")
        self.pattern4 = re.compile("(where.*?[?])")
        self.pattern5 = re.compile("(who.*?[?])")
        self.pattern6 = re.compile("(does.*?[?])")

        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            count1 = len(re.findall(self.pattern1, ex))
            count2 = len(re.findall(self.pattern2, ex))
            count3 = len(re.findall(self.pattern3, ex))
            count4 = len(re.findall(self.pattern4, ex))
            count5 = len(re.findall(self.pattern5, ex))
            count6 = len(re.findall(self.pattern6, ex))
            count= count1 + count3 + count2 + count4+count5+count6

            features[i, 0] =count
            i += 1

        return features

# TODO: Add custom feature transformers for the movie review data


class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer())
            ])),
            ('question_marks', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('question_marks', starts_with_questioning())
            ])),
            ('year_mentioned', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('year', year_mentioned())
            ])),
            # ('negation_words', Pipeline([
            #     ('selector', ItemSelector(key='text')),
            #     ('negative', negative_sense())
            # ])),

            # ('ngram',Pipeline([
            #     ('selector', ItemSelector(key='text')),
            #     ('vect', TfidfVectorizer(ngram_range=(1,1),token_pattern="([a-zA-Z]+|[?]+|[!]+|[.]+)\S*")),
            #
            # ])),
            ('one_gram', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('vect_count', CountVectorizer(ngram_range=(1, 1), token_pattern="([a-zA-Z]+|[?]+|[!]+|[.]+)\S*")),

            ])),

        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    print("Label set: %s\n" % str(labels))

    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    #print(feat_train)
    #print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, max_iter=15000, shuffle=True, verbose=2)

    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)

    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier
