import csv
import logging

from matplotlib import pyplot as plt
import numpy as np

from src.features.build_features import *
from src.constants import *


def visualize_photon_count(*Xs, max_photon_count, filename=None):
    """
    Display plot of Xs[0]'s photon counts stacked on top of Xs[1]'s...
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title("Distribution of Photon Count")
    ax.set_xlabel("Number of Photons")
    ax.set_ylabel("Number of Qubits")
    frequencies = []
    bottom = np.array([0 for i in range(0, max_photon_count + 1)])
    for index, X in enumerate(Xs):
        freq = {}
        for x in X:
            freq[len(x)] = freq.get(len(x), 0) + 1

        frequencies.append(np.array([freq.get(i, 0) for i in range(0, max_photon_count + 1)]))

        plt.bar(
            x=list(range(0, max_photon_count + 1)), 
            height=frequencies[index],
            bottom=bottom,
            label=index)

        bottom += frequencies[len(frequencies) - 1]
    plt.legend()
    
    if filename == None:
        plt.show()
    else:
        plt.savefig("./reports/figures/" + filename + ".png")

def visualize_photon_count_group_by_traintest_truth(X, y, indices):
    X_train, y_train = X[indices[0][0]], y[indices[0][0]]
    X_test, y_test = X[indices[0][1]], y[indices[0][1]]
    X_train_p, y_train_p, X_train_n, y_train_n, _, _, _, _ = filter_datapoints(X_train, y_train, y_train)
    X_test_p, y_test_p, X_test_n, y_n, _, _, _, _ = filter_datapoints(X_test, y_test, y_test)
    visualize_photon_count(X_train_p, X_train_n, X_test_p, X_test_n)

def visualize_cumulative_arrival_distribution(*Xs, first_arrival, last_arrival, filename=None):
    """
    Displays plot of Xs[0]'s photon arrival time distributions adjacent to Xs[1]'s...
    """
    fig, axs = plt.subplots(len(Xs), figsize=FIG_SIZE)
    fig.suptitle("Distribution of Photons' Arrival Times")
    for i in range(len(Xs)):
        if len(Xs) == 1:
            ax = axs
        else:
            ax = axs[i]

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Number of Photons")
        ax.set_xticks(np.linspace(first_arrival, last_arrival, 18))
        ax.label_outer()

        ax.hist([photon for row in Xs[i] for photon in row], bins=2000, range=(first_arrival, last_arrival))
    
    if filename == None:
        plt.show()
    else:
        plt.savefig("./reports/figures/" + filename + ".png")

def visualize_individual_arrival_distribution(X, limit, first_arrival, last_arrival, filename=None):
    """
    Display up to <<limit>> plots at a time of each datapoint's photon arrival time distributions
    Keep displaying plots until all datapoints have been visualized
    """
    count = 0
    while count < len(X):
        fig, axs = plt.subplots(min(limit, len(X) - count), figsize=FIG_SIZE)
        fig.suptitle("Distribution of Photons' Arrival Times")
        for i in range(min(limit, len(X) - count)):
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("Number of Photons")
            axs[i].set_xticks(np.linspace(first_arrival, last_arrival, 20))
            axs[i].set_yticks(np.linspace(0, 2, 3))
            axs[i].label_outer()

            axs[i].hist(X[count], bins=1000, range=(first_arrival, last_arrival))
            count += 1

        if filename == None:
            plt.show()
        else:
            fig.savefig("./reports/figures/" + filename + str(count) + ".png")

if __name__ == '__main__':
    # X, y = load_data()
    # fp_instances, fn_instances = load_classifier_test_results(
            # ['classifier_test_result_mlp_{}.csv'.format(n) for n in range(0, 5)]
            # + ['classifier_test_result_mlp_kfold_{}.csv'.format(n) for n in range(0, 5)])
            # ['Results/falsely-classified-instances-mlp-lg-rf/classifier_test_result_{}.csv'.format(n) for n in range(0, 15)]
            # ['./data/interim/classifier_test_result_{}.csv'.format(n) for n in range(0, 1)])
    # visualize_photon_count(fp_instances, fn_instances)
    # write_instances_to_file(X, fp_indices, fn_indices)
    # logging.info("Done.")
    pass
