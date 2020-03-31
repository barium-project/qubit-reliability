from os import sys
import csv
import logging
import numpy as np
from matplotlib import pyplot as plt

from src.features.build_features import *

def use_ramdisk(filenames):
    return list(map(lambda filename: "/Volumes/ramdisk/" + filename, filenames))

def draw_plot(qubits_measurements):
    fig, ax = plt.subplots()
    ax.set_title("Distribution of Photons' Arrival Times")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Photons")
    ax.set_xticks(np.linspace(0, 0.0051, 18))
    plt.hist(qubits_measurements, bins=2000, range=(0, 0.0051))
    plt.show()


if __name__ == '__main__':
    X, y = load_data()
    draw_plot([photon for row in X for photon in row])
    logging.info("Done.")
