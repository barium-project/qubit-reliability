import csv
from enum import Enum
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

import numpy as np

QUBIT_DATASET = [
    ['./data/processed/v1/BrightTimeTagSet1.csv',
    './data/processed/v1/BrightTimeTagSet2.csv',
    './data/processed/v1/BrightTimeTagSet3.csv',
    './data/processed/v1/BrightTimeTagSet4.csv',
    './data/processed/v1/BrightTimeTagSet5.csv',],
    ['./data/processed/v1/DarkTimeTagSet1.csv',
    './data/processed/v1/DarkTimeTagSet2.csv',
    './data/processed/v1/DarkTimeTagSet3.csv',
    './data/processed/v1/DarkTimeTagSet4.csv',
    './data/processed/v1/DarkTimeTagSet5.csv',]]

def load_data():
    X, y = [], []

    for i in [0, 1]:
        for file_name in QUBIT_DATASET[i]:
            logging.info("Loading {}".format(file_name))
            with open(file_name, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    X.append([float(photon) for photon in row])
                    y.append(i)

    return X, y

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
