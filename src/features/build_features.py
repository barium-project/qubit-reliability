import csv
import logging
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import *


class Histogramizer(BaseEstimator, TransformerMixin):
    def __init__(self, bins, range):
        super().__init__()
        self.range = range
        self.bins = bins
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return list(map(lambda x: np.histogram(x, bins=self.bins, range=self.range)[0], X))


def load_data(dataset, stats):
    X, y = [], []
    s = {'first_arrival': 1.0, 'last_arrival': 0.0, 'max_count': 0, 'file_range': {}}

    for i in [0, 1]:
        for file_name in QUBIT_DATASET[dataset][i]:
            logging.info("Loading {}".format(file_name))
            with open(file_name, 'r') as file:
                s['file_range'][file_name] = [len(X), 0]
                reader = csv.reader(file)
                for row in reader:
                    row = [float(photon) for photon in row]
                    if len(row) > 0 and stats:
                        s['first_arrival'] = min(s['first_arrival'], min(row))
                        s['last_arrival'] = max(s['last_arrival'], max(row))
                        s['max_count'] = max(s['max_count'], len(row))
                    X.append(row)
                    y.append(i)
                s['file_range'][file_name][1] = len(X) - 1

    if stats:
        return np.array(X), np.array(y), s
    else:
        return np.array(X), np.array(y)

def filter_datapoints(X, y, y_pred, indices=[]):
    """
    Filter datapoints into positive, negative, false positive, false negative
    """
    X_n, y_n, i_n = [], [], []
    X_p, y_p, i_p = [], [], []
    X_fp, y_fp, i_fp = [], [], []
    X_fn, y_fn, i_fn = [], [], []

    for i in range(0, len(X)):
        if y[i] == 0 and y_pred[i] == 0:
            X_n.append(X[i])
            y_n.append(y[i])
            if len(indices) > 0:
                i_n.append(indices[i])
        elif y[i] == 1 and y_pred[i] == 1:
            X_p.append(X[i])
            y_p.append(y[i])
            if len(indices) > 0:
                i_p.append(indices[i])
        elif y[i] == 0 and y_pred[i] == 1:
            X_fp.append(X[i])
            y_fp.append(y[i])
            if len(indices) > 0:
                i_fp.append(indices[i])
        elif y[i] == 1 and y_pred[i] == 0:
            X_fn.append(X[i])
            y_fn.append(y[i])
            if len(indices) > 0:
                i_fn.append(indices[i])

    return {'X_n': np.array(X_n), 
            'y_n': np.array(y_n), 
            'i_n': np.array(i_n),
            'X_p': np.array(X_p), 
            'y_p': np.array(y_p), 
            'i_p': np.array(i_p),
            'X_fp': np.array(X_fp), 
            'y_fp': np.array(y_fp), 
            'i_fp': np.array(i_fp),
            'X_fn': np.array(X_fn), 
            'y_fn': np.array(y_fn),
            'i_fn': np.array(i_fn)}


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
