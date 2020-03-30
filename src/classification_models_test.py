import csv
import logging
from abc import ABC, abstractmethod
import sys
import numpy as np
import matplotlib.pyplot as plt

from features.build_features import *


BEST_RELIABILITY_ACHIEVED = 0.9997081106493118


class ClassificationModel(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def classify(self, qubit_measurement):
        pass


class ThresholdCutoffModel(ClassificationModel):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def __str__(self):
        return "Threshold Cutoff Model w/ threshold {}".format(self.threshold)

    def classify(self, x):
        return 0 if len(x) > self.threshold else 1

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.classify(x))
        return y_pred

    def evaluate(self, X, y):
        n = len(y)
        y_ground_pred = list(zip(*[y, self.predict(X)]))
        false_positives = len([ground_pred for ground_pred in y_ground_pred if ground_pred[0] == 0 and ground_pred[1] == 1])
        false_negatives = len([ground_pred for ground_pred in y_ground_pred if ground_pred[0] == 1 and ground_pred[1] == 0])
        accuracy = 1 - (float(false_positives + false_negatives) / n)
        
        return accuracy, float(false_positives) / n, float(false_negatives) / n


class ThresholdCutoffEarlyArrivalModel(ClassificationModel):
    def __init__(self, threshold, arrival_time):
        super().__init__()
        self.threshold = threshold
        self.arrival_time = arrival_time

    def __str__(self):
        return "Threshold Cutoff Early Arrival Model w/ photon number threshold {} and arrival time threshold {}".format(
            self.threshold, self.arrival_time)

    def classify(self, x):
        new_x = list(filter(lambda photon: photon < self.arrival_time, x))
        return 0 if len(new_x) > self.threshold else 1

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.classify(x))
        return y_pred

    def evaluate(self, X, y):
        n = len(y)
        y_ground_pred = list(zip(*[y, self.predict(X)]))
        false_positives = len([ground_pred for ground_pred in y_ground_pred if ground_pred[0] == 0 and ground_pred[1] == 1])
        false_negatives = len([ground_pred for ground_pred in y_ground_pred if ground_pred[0] == 1 and ground_pred[1] == 0])
        accuracy = 1 - (float(false_positives + false_negatives) / n)
        
        return accuracy, float(false_positives) / n, float(false_negatives) / n

def threshold_cutoff_experiments():
    X, y = read_qubit_measurements()

    most_photons_received = max(map(lambda row: len(row), X))
    print("Max number of photons captured for one qubit: {}".format(most_photons_received))

    res = []
    # try to classify measurements with a range of cutoff values and look at their accuracy
    # for threshold in range(12, 13):
    for threshold in range(0, most_photons_received + 1):
        model = ThresholdCutoffModel(threshold)
        accuracy, false_positive, false_negative = model.evaluate(X, y)
        res.append((threshold, accuracy))
    
    print("Threshold Cutoff Model Accuracy:")
    for threshold, accuracy in res:
        print("{},{}".format(threshold, accuracy))


def find_false_classifications_with_photon_histogram(limit):
    """
    Classify qubits by the Threshold Cutoff Model with the optimal threshold, find all mis-classified qubits and
    print the histogram of each's measured photons (frequency of every arriving time interval)
    """
    X, y = read_qubit_measurements()
    model = ThresholdCutoffModel(12)
    y_ground_pred = list(zip(*[y, model.predict(X)]))

    false_positive_qubits = list(filter(lambda ground_pred: ground_pred[0] == 0 and ground_pred[1] == 1, y_ground_pred))
    fig, axs = plt.subplots(limit if limit > 0 else len(false_positive_qubits))
    fig.suptitle("Distribution of Photons' Arrival Times")
    count = 0
    for i, ground_pred in enumerate(y_ground_pred):
        if ground_pred[0] == 0 and ground_pred[1] == 1:
            axs[count].set_xlabel("Time")
            axs[count].set_ylabel("Number of Photons")
            axs[count].set_xticks(np.linspace(0, 0.0051, 18))
            axs[count].set_yticks(np.linspace(0, 2, 3))
            axs[count].hist(X[i], bins=1000, range=(0, 0.0051))
            print(X[i])

            count += 1

            if limit > 0 and count == limit:
                break

    for ax in axs.flat:
        ax.label_outer()
    plt.show()

    false_negative_qubits = list(filter(lambda ground_pred: ground_pred[0] == 1 and ground_pred[1] == 0, y_ground_pred))
    fig, axs = plt.subplots(limit if limit > 0 else len(false_negative_qubits))
    fig.suptitle("Distribution of Photons' Arrival Times")
    count = 0
    for i, ground_pred in enumerate(y_ground_pred):
        if ground_pred[0] == 1 and ground_pred[1] == 0:
            axs[count].set_xlabel("Time")
            axs[count].set_ylabel("Number of Photons")
            axs[count].set_xticks(np.linspace(0, 0.0051, 18))
            axs[count].set_yticks(np.linspace(0, 2, 3))
            axs[count].hist(X[i], bins=1000, range=(0, 0.0051))
            print(X[i])

            count += 1

            if limit > 0 and count == limit:
                break

    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def threshold_cutoff_early_arrival_experiments():
    """
    Per the idea in paper "Machine learning assisted readout of trapped-ion qubits",
    try filter out late arrival photons in the Threshold Cutoff classification approach
    """
    with open('../data/interim/threshold_cutoff_early_arrival_experiment.csv', 'w') as result_file:
        writer = csv.writer(result_file)

        X, y = read_qubit_measurements()
        most_photons_received = max(map(lambda row: len(row), X))
        latest_photon_arrival_time = max(map(lambda row: max(row) if len(row) > 0 else 0, X))
        logging.info("Latest photon arrival time among all measurements: {}".format(latest_photon_arrival_time))

        for arrival_time in np.arange(latest_photon_arrival_time, 0, -0.0001):
            for threshold in range(12, 13):
            #for threshold in range(1, 41):  # thresholds that achieve accuracy > 70% without early arrival model
                model = ThresholdCutoffEarlyArrivalModel(threshold, arrival_time)
                logging.info("Testing {}".format(model))
                accuracy, false_positives, false_negatives = model.evaluate(X, y)

                writer.writerow([arrival_time, threshold, accuracy, false_positives, false_negatives])
                result_file.flush()
                if accuracy > BEST_RELIABILITY_ACHIEVED:
                    print("Higher Accuracy Achieved! Accuracy = {}, Model: {}".format(accuracy, model))


if __name__ == '__main__':
    threshold_cutoff_experiments()
    # find_false_classifications_with_photon_histogram(limit=10)
    # threshold_cutoff_early_arrival_experiments()
