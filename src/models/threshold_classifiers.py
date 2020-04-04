import csv
import logging
import sys
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from src.features.build_features import *
from src.util import *
from src.constants import *


class ThresholdCutoffClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def fit(self, X, y):
        return self

    def classify(self, x):
        return 0 if len(x) > self.threshold else 1

    def predict(self, X):
        return [self.classify(x) for x in X]

class ThresholdCutoffEarlyArrivalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold, arrival_time):
        super().__init__()
        self.threshold = threshold
        self.arrival_time = arrival_time

    def fit(self, X, y):
        return self

    def classify(self, x):
        new_x = list(filter(lambda photon: photon < self.arrival_time, x))
        return 0 if len(new_x) > self.threshold else 1

    def predict(self, X):
        return [self.classify(x) for x in X]

def threshold_cutoff_early_arrival_experiments():
    """
    Per the idea in paper "Machine learning assisted readout of trapped-ion qubits",
    try filter out late arrival photons in the Threshold Cutoff classification approach
    """
    with open('./data/interim/threshold_cutoff_early_arrival_experiment.csv', 'w') as result_file:
        writer = csv.writer(result_file)

        X, y = load_data()
        most_photons_received = max(map(lambda row: len(row), X))
        latest_photon_arrival_time = max(map(lambda row: max(row) if len(row) > 0 else 0, X))
        logging.info("Latest photon arrival time among all measurements: {}".format(latest_photon_arrival_time))

        for arrival_time in np.arange(latest_photon_arrival_time, 0, -0.0001):
            for threshold in range(12, 13):
            #for threshold in range(1, 41):  # thresholds that achieve accuracy > 70% without early arrival model
                model = ThresholdCutoffEarlyArrivalClassifier(threshold, arrival_time)
                logging.info("Testing {}".format(model))
                accuracy, false_positives, false_negatives = model.evaluate(X, y)

                writer.writerow([arrival_time, threshold, accuracy, false_positives, false_negatives])
                result_file.flush()
                if accuracy > 0.9997081106493118:
                    print("Higher Accuracy Achieved! Accuracy = {}, Model: {}".format(accuracy, model))


if __name__ == '__main__':
    threshold_cutoff_early_arrival_experiments()
