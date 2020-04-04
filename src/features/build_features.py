import csv
import logging
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import *


class Histogramizer(BaseEstimator, TransformerMixin):
    def __init__(self, bins, range=(FIRST_ARRIVAL, LAST_ARRIVAL)):
        super().__init__()
        self.range = range
        self.bins = bins
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return list(map(lambda x: np.histogram(x, bins=self.bins, range=self.range)[0], X))


def load_data():
    X, y = [], []
    min_time = 1.0
    max_time = 0.0
    for i in [0, 1]:
        for file_name in QUBIT_DATASET["V3"][i]:
            logging.info("Loading {}".format(file_name))
            with open(file_name, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len([float(photon) for photon in row]) > 0:
                        min_time = min(min_time, min([float(photon) for photon in row]))
                        max_time = max(max_time, max([float(photon) for photon in row]))
                    X.append([float(photon) for photon in row])
                    y.append(i)
    print(min_time)
    print(max_time)

    return np.array(X), np.array(y)

def filter_datapoints(X, y, y_pred):
    """
    Filter datapoints into positive, negative, false positive, false negative
    """
    X_p, y_p = [], []
    X_n, y_n = [], []
    X_fp, y_fp = [], []
    X_fn, y_fn = [], []

    for i in range(0, len(X)):
        if y[i] == 0 and y_pred[i] == 0:
            X_n.append(X[i])
            y_n.append(y[i])
        elif y[i] == 1 and y_pred[i] == 1:
            X_p.append(X[i])
            y_p.append(y[i])
        elif y[i] == 0 and y_pred[i] == 1:
            X_fp.append(X[i])
            y_fp.append(y[i])
        elif y[i] == 1 and y_pred[i] == 0:
            X_fn.append(X[i])
            y_fn.append(y[i])

    return np.array(X_p), np.array(y_p), np.array(X_n), np.array(y_n), np.array(X_fp), np.array(y_fp), np.array(X_fn), np.array(y_fn)

def load_classifier_test_results(filenames):
    fp_instances = []  # false positives
    fn_instances = []  # false negatives
    for result_filename in filenames:
        with open(result_filename, 'r') as result_file:
            logging.info("Loading {}".format(result_filename))
            csv_reader = csv.reader(result_file)
            for line in csv_reader:
                if line[0] == 'FALSE_POSITIVE':
                    fp_instances.append(
                        list(map(lambda timestamp: float(timestamp), line[1:])))
                if line[0] == 'FALSE_NEGATIVE':
                    fn_instances.append(
                        list(map(lambda timestamp: float(timestamp), line[1:])))
    return fp_instances, fn_instances

if __name__ == "__main__":
    pass
