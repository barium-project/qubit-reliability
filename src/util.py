import csv
import logging
from os import path

import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.constants import *

class Histogramize(BaseEstimator, TransformerMixin):
    def __init__(self, num_buckets=6, arrival_time_threshold=(FIRST_ARRIVAL, LAST_ARRIVAL)):
        self.num_buckets = num_buckets
        self.arrival_time_threshold = arrival_time_threshold
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        histogram_bins = np.linspace(
            self.arrival_time_threshold[0], self.arrival_time_threshold[1], 
            num=(self.num_buckets+1), endpoint=True)
        return list(map(
            lambda measurement: np.histogram(measurement, bins=histogram_bins)[0], X))

def classifier_train(classifier, qubits_measurements_train, qubits_truths_train):
    logging.info("Training Classifier: {}".format(classifier))
    classifier.fit(qubits_measurements_train, qubits_truths_train)
    return classifier

_classifier_test_counter = 0
CLASSIFIER_TEST_OUTPUT_FILENAME_BASE = 'classifier_test_result'

def classifier_test(classifier, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test, test_training=False):
    global _classifier_test_counter
    logging.info("Testing classifier: {}".format(classifier))

    if test_training:
        qubits_predict_train = classifier.predict(qubits_measurements_train)
    qubits_predict_test = classifier.predict(qubits_measurements_test)

    if test_training:
        print("Classification Report on Training Data:")
        print(confusion_matrix(qubits_truths_train, qubits_predict_train))
        print(classification_report(qubits_truths_train, qubits_predict_train, digits=8))

    print("Classification Report on Testing Data:")
    print(confusion_matrix(qubits_truths_test, qubits_predict_test))
    print(classification_report(qubits_truths_test, qubits_predict_test, digits=8))

    # Print out all instances of false positives and false negatives in the test set
    if test_training:
        assert(len(qubits_measurements_train) == len(qubits_truths_train) == len(qubits_predict_train))
    assert(len(qubits_measurements_test) == len(qubits_truths_test) == len(qubits_predict_test))
    false_positives_test = list(map(
        lambda index: qubits_measurements_test[index], 
        list(filter(
            lambda index: qubits_truths_test[index] == 0 and qubits_predict_test[index] == 1, 
            range(len(qubits_measurements_test))))))
    false_negatives_test = list(map(
        lambda index: qubits_measurements_test[index], 
        list(filter(
            lambda index: qubits_truths_test[index] == 1 and qubits_predict_test[index] == 0, 
            range(len(qubits_measurements_test))))))

    output_filename = "./data/interim/{base}_{counter}.csv".format(
        base=CLASSIFIER_TEST_OUTPUT_FILENAME_BASE, counter=_classifier_test_counter)
    with open(output_filename, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([str(classifier)])
        for instance in false_positives_test:
            csv_writer.writerow(["FALSE_POSITIVE"] + list(instance))
        for instance in false_negatives_test:
            csv_writer.writerow(["FALSE_NEGATIVE"] + list(instance))

    _classifier_test_counter += 1
    logging.info("Falsely-classified instances written to the report file.")

    return accuracy_score(qubits_truths_test, qubits_predict_test)

def picklize(db_id, overwrite=False):
    def decorator(function):
        def wrapper(*args, **kwargs):
            def _pickle_db_path(db_id):
                return "./data/interim/{}.pickle".format(db_id)

            db_filename = _pickle_db_path(db_id)
            if (not overwrite) and path.exists(db_filename):
                logging.info("Pickle: Loading from database {}".format(db_filename))
                with open(db_filename, 'rb') as db_file:
                    return pickle.load(db_file)
            else:
                ret = function(*args, **kwargs)
                with open(db_filename, 'wb') as db_file:
                    pickle.dump(ret, db_file)
                return ret
        return wrapper
    return decorator


if __name__ == '__main__':
    pass
