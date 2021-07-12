from __future__ import print_function
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import numpy as np


class Classifier(object):

    def __init__(self, vectors):
        self.embeddings = vectors

    def __call__(self, train_index, test_index, val_index, Y, seed=0):

        numpy.random.seed(seed)

        averages = ["micro", "macro"] #
        f1s = {}

        # Y = np.argmax(Y, -1)


        X_train = [self.embeddings[x] for x in train_index]
        Y_train = [Y[x] for x in train_index]
        X_test = [self.embeddings[x] for x in test_index]
        Y_test = [Y[x] for x in test_index]

        clf = LogisticRegression()
        clf.fit(X_train, Y_train)
        Y_  = clf.predict(X_test)

        for average in averages:
            f1s[average]= f1_score(Y_test, Y_, average=average)
        acc = accuracy_score(Y_test, Y_)
        return acc, f1s["micro"], f1s["macro"]





