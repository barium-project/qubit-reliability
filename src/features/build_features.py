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

def load_datasets():
    qubits_measurements = []
    for dataset_filename in BRIGHT_QUBITS_DATASETS + DARK_QUBITS_DATASETS:
        with open(dataset_filename, 'r') as dataset_file:
            logging.info("Loading {}".format(dataset_filename))
            csv_reader = csv.reader(dataset_file)
            for line in csv_reader:
                qubits_measurements.append(
                    list(map(lambda timestamp: float(timestamp), line)))
    return qubits_measurements

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

def load_datasets2():
    def load_datasets_with_ground_truth(qubits_datasets, ground_truth):
        qubits_measurements = []
        for dataset_filename in qubits_datasets:
            with open(dataset_filename, 'r') as dataset_file:
                logging.info("Loading {}".format(dataset_filename))
                csv_reader = csv.reader(dataset_file)
                for line in csv_reader:
                    qubits_measurements.append(
                        np.array(list(map(lambda timestamp: float(timestamp), line)))
                    )
        qubits_ground_truths = [ground_truth for i in range(len(qubits_measurements))]
        return qubits_measurements, qubits_ground_truths
    
    bright_qubits_measurements, bright_qubits_ground_truths = load_datasets_with_ground_truth(BRIGHT_QUBITS_DATASETS, 0)
    dark_qubits_measurements, dark_qubits_ground_truths = load_datasets_with_ground_truth(DARK_QUBITS_DATASETS, 1)
    return (
        (bright_qubits_measurements + dark_qubits_measurements), 
        (bright_qubits_ground_truths + dark_qubits_ground_truths))

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
