import csv
from enum import Enum
import logging

import numpy as np

BRIGHT_QUBITS_DATASETS = [
    '../data/processed/Data4Jens/BrightTimeTagSet1.csv',
    '../data/processed/Data4Jens/BrightTimeTagSet2.csv',
    '../data/processed/Data4Jens/BrightTimeTagSet3.csv',
    '../data/processed/Data4Jens/BrightTimeTagSet4.csv',
    '../data/processed/Data4Jens/BrightTimeTagSet5.csv',
]

DARK_QUBITS_DATASETS = [
    '../data/processed/Data4Jens/DarkTimeTagSet1.csv',
    '../data/processed/Data4Jens/DarkTimeTagSet2.csv',
    '../data/processed/Data4Jens/DarkTimeTagSet3.csv',
    '../data/processed/Data4Jens/DarkTimeTagSet4.csv',
    '../data/processed/Data4Jens/DarkTimeTagSet5.csv',
]

class Qubit(Enum):
    BRIGHT = 0
    DARK = 1

class QubitMeasurement():
    def __init__(self, photons, ground_truth):
        super().__init__()
        self.photons = photons
        self.ground_truth = ground_truth
        self.classified_result = None

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
    def read_from_files_with_ground_truth(filenames, ground_truth, qubit_measurements):
        for measurement_filename in filenames:
            logging.info("Loading {}".format(measurement_filename))
            with open(measurement_filename, 'r') as measurement_file:
                reader = csv.reader(measurement_file)
                for photons in reader:
                    qubit_measurements.append(QubitMeasurement([float(photon) for photon in photons], ground_truth))
        return qubit_measurements

    qubit_measurements = []
    read_from_files_with_ground_truth(BRIGHT_QUBITS_DATASETS, Qubit.BRIGHT, qubit_measurements)
    read_from_files_with_ground_truth(DARK_QUBITS_DATASETS, Qubit.DARK, qubit_measurements)
    return qubit_measurements

def load_datasets(filenames):
    qubits_measurements = []
    for dataset_filename in filenames:
        with open(dataset_filename, 'r') as dataset_file:
            logging.info("Loading {}".format(dataset_filename))
            csv_reader = csv.reader(dataset_file)
            for line in csv_reader:
                qubits_measurements.extend(
                    list(map(lambda timestamp: float(timestamp), line)))
    return qubits_measurements
