from os import sys
import csv
import logging
import numpy as np
from matplotlib import pyplot as plt

from features.build_features import *

def use_ramdisk(filenames):
    return list(map(lambda filename: "/Volumes/ramdisk/" + filename, filenames))

def draw_plot(qubits_measurements):
    logging.info("Plotting histogram graph.")
    fig, ax = plt.subplots()
    ax.set_title("Distribution of Photons' Arrival Times")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Photons")
    ax.set_xticks(np.linspace(0, 0.006, 21))
    # plt.hist(qubits_measurements, bins=2000)
    plt.hist(qubits_measurements, bins=1000)
    plt.show()


if __name__ == '__main__':
    qubits_measurements = load_datasets(BRIGHT_QUBITS_DATASETS + DARK_QUBITS_DATASETS)
    # qubits_measurements = load_datasets([
    #     'Results/falsely-classified-instances-mlp-lg-rf/false_positive_instances.csv', 
    #     'Results/falsely-classified-instances-mlp-lg-rf/false_negative_instances.csv'
    # ])
    draw_plot(qubits_measurements)
    logging.info("Done.")
