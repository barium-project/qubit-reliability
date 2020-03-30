import csv
from enum import Enum
import logging

import numpy as np

BRIGHT_QUBITS_DATASETS = [
    '../data/processed/v1/BrightTimeTagSet1.csv',
    '../data/processed/v1/BrightTimeTagSet2.csv',
    '../data/processed/v1/BrightTimeTagSet3.csv',
    '../data/processed/v1/BrightTimeTagSet4.csv',
    '../data/processed/v1/BrightTimeTagSet5.csv',
]

DARK_QUBITS_DATASETS = [
    '../data/processed/v1/DarkTimeTagSet1.csv',
    '../data/processed/v1/DarkTimeTagSet2.csv',
    '../data/processed/v1/DarkTimeTagSet3.csv',
    '../data/processed/v1/DarkTimeTagSet4.csv',
    '../data/processed/v1/DarkTimeTagSet5.csv',
]

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

def read_qubit_measurements():
    X = []
    y = []

    for file_name in BRIGHT_QUBITS_DATASETS:
        logging.info("Loading {}".format(file_name))
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                X.append([float(photon) for photon in row])
                y.append(0)

    for file_name in DARK_QUBITS_DATASETS:
        logging.info("Loading {}".format(file_name))
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                X.append([float(photon) for photon in row])
                y.append(1)

    return X, y

if __name__ == "__main__":
    pass
