import pandas as pd
import numpy as np
import csv
import logging
from os import path, sys
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from src.features.build_features import *
from src.util import *


class ThresholdCutoffClassifier(BaseEstimator, ClassifierMixin):
    """
    Given a histogram of photons' arrival times as the data instance, 
    classify the qubit by counting the number of captured photons
    """
    def __init__(self, threshold=BEST_PHOTON_COUNT_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y):
        return self

    def predict(self, X):
        return list(map(lambda instance: 0 if sum(instance) > self.threshold else 1, X))


# Logistic Regression
def logistic_regression_grid_search_cv(qubits_measurements_train, qubits_truths_train):
    logging.info("Starting Grid Search with Cross Validation on Logistic Regression models.")

    lg_pipeline = Pipeline([
        ('hstgm', Histogramize(num_buckets=6)),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])

    lg_param_grid = {
        # 'histogram__num_buckets': range(2, 33),
        'clf__penalty': ['none', 'l1', 'l2'],
        'clf__C': [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]
    }

    lg_grid = GridSearchCV(lg_pipeline, cv=4, n_jobs=-1, param_grid=lg_param_grid, scoring="accuracy", refit=True, verbose=2)
    lg_grid.fit(qubits_measurements_train, qubits_truths_train)
    return lg_grid


# Random Forest
def random_forest_grid_search_cv(qubits_measurements_train, qubits_truths_train):
    logging.info("Starting Grid Search with Cross Validation on Random Forest Classifier.")
    
    rf_pipeline = Pipeline([
        ('hstgm', Histogramize(num_buckets=6)),
        ('clf', RandomForestClassifier())
    ])

    rf_param_grid = {}

    rf_grid = GridSearchCV(rf_pipeline, cv=4, n_jobs=-1, param_grid=rf_param_grid, scoring="accuracy", verbose=2)
    rf_grid.fit(qubits_measurements_train, qubits_truths_train)
    return rf_grid


# Majority Vote with improved Feed-forward Neural Network
def majority_vote_grid_search_cv(qubits_measurements_train, qubits_truths_train, **kwargs):
    cv = kwargs['cv'] if 'cv' in kwargs else 4
    logging.info(cv)

    logging.info("Starting Grid Search with Cross Validation on Majority Vote Classifier.")

    mv_pipeline = Pipeline([
        ('hstgm', Histogramize(
            arrival_time_threshold=(PRE_ARRIVAL_TIME_THRESHOLD, POST_ARRIVAL_TIME_THRESHOLD))),
        ('clf', VotingClassifier([
            ('mlp', 
                MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(32, 32), random_state=RANDOM_SEED)),
            ('tc', 
                ThresholdCutoffClassifier(threshold=BEST_PHOTON_COUNT_THRESHOLD)),
            ('lg', 
                LogisticRegression(solver='liblinear', penalty='l2', C=10**-3, random_state=RANDOM_SEED)),
            ('rf', 
                RandomForestClassifier(random_state=RANDOM_SEED))
        ]))
    ])

    mv_param_grid = {
        'hstgm__num_buckets': range(5, 10),
        'clf__mlp__hidden_layer_sizes': [(neurons,) * 2 for neurons in range(20, 41)],
    }

    mv_grid = GridSearchCV(mv_pipeline, cv=cv, n_jobs=-1, param_grid=mv_param_grid, scoring="accuracy", refit=True, verbose=2)
    mv_grid.fit(qubits_measurements_train, qubits_truths_train)
    return mv_grid


# Main Tasks
def run_logistic_regression_with_kfold_data_split():
    """
    Logistic Regression with the best parameters found before, using 5-fold data split
    Mostly concerned with the falsely-classified instances fed to further analysis
    """
    qubits_measurements, qubits_truths = load_data()
    qubits_measurements = np.array(qubits_measurements)
    qubits_truths = np.array(qubits_truths)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf_accuracies = []
    _i_fold = 0
    for train_index, test_index in kf.split(qubits_measurements):
        _i_fold += 1
        logging.info("Train/Test data split {fold}-th fold.".format(fold=_i_fold))

        qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
            qubits_measurements[train_index], qubits_measurements[test_index], \
            qubits_truths[train_index], qubits_truths[test_index]

        lg_pipeline = classifier_train(Pipeline([
                ('hstgm', Histogramize(num_buckets=6, arrival_time_threshold=(0, BEST_ARRIVAL_TIME_THRESHOLD))),
                ('clf', LogisticRegression(solver='liblinear', random_state=RANDOM_SEED, penalty='l2', C=10**-3))
            ]), qubits_measurements_train, qubits_truths_train)

        curr_accuracy = classifier_test(lg_pipeline, qubits_measurements_train, qubits_measurements_test, 
                qubits_truths_train, qubits_truths_test)
        print("Train/test split {fold}-th fold accuracy: {accuracy}".format(fold=_i_fold, accuracy=curr_accuracy))
        clf_accuracies.append(curr_accuracy)
    
    avg_accuracy = sum(clf_accuracies) / len(clf_accuracies)
    print("Logistic Regression with KFold Data Split: Average Accuracy = {accuracy}".format(accuracy=avg_accuracy))


def run_logistic_regression_grid_search_cv():
    qubits_measurements, qubits_truths = load_data()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)

    lg_grid = picklize('logistic_regression_grid_search_cv') \
        (logistic_regression_grid_search_cv) \
        (qubits_measurements_train, qubits_truths_train)

    classifier_test(lg_grid, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test)


def run_random_forest_with_kfold_data_split():
    """
    Random forest with the best parameters found before (currently no parameter), using 5-fold data split
    Mostly concerned with the falsely-classified instances fed to further analysis
    """
    qubits_measurements, qubits_truths = load_data()
    qubits_measurements = np.array(qubits_measurements)
    qubits_truths = np.array(qubits_truths)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf_accuracies = []
    _i_fold = 0
    for train_index, test_index in kf.split(qubits_measurements):
        _i_fold += 1
        logging.info("Train/Test data split {fold}-th fold.".format(fold=_i_fold))

        qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
            qubits_measurements[train_index], qubits_measurements[test_index], \
            qubits_truths[train_index], qubits_truths[test_index]

        rf_pipeline = classifier_train(Pipeline([
                ('hstgm', Histogramize(num_buckets=6, arrival_time_threshold=(0, BEST_ARRIVAL_TIME_THRESHOLD))),
                ('clf', RandomForestClassifier())
            ]), qubits_measurements_train, qubits_truths_train)

        curr_accuracy = classifier_test(rf_pipeline, qubits_measurements_train, qubits_measurements_test, 
                qubits_truths_train, qubits_truths_test)
        print("Train/test split {fold}-th fold accuracy: {accuracy}".format(fold=_i_fold, accuracy=curr_accuracy))
        clf_accuracies.append(curr_accuracy)
    
    avg_accuracy = sum(clf_accuracies) / len(clf_accuracies)
    print("Random Forest with KFold Data Split: Average Accuracy = {accuracy}".format(accuracy=avg_accuracy))


def run_random_forest_grid_search_cv():
    qubits_measurements, qubits_truths = load_data()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)
        
    rf_grid = picklize('random_forest_grid_search_cv') \
        (random_forest_grid_search_cv)(qubits_measurements_train, qubits_truths_train)
    logging.info(pd.DataFrame(rf_grid.cv_results_))

    classifier_test(rf_grid, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test)


def run_majority_vote_with_kfold_data_split():
    """
    Split the dataset 5 times into 80% training and 20% testing, 
    perform Majority Vote classification on each set, and report
    the average accuracy across the 5 folds.
    """
    qubits_measurements, qubits_truths = load_data()
    qubits_measurements = np.array(qubits_measurements)
    qubits_truths = np.array(qubits_truths)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf_accuracies = []
    _i_fold = 0
    for train_index, test_index in kf.split(qubits_measurements):
        _i_fold += 1
        logging.info("Train/Test data split {fold}-th fold.".format(fold=_i_fold))

        qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
            qubits_measurements[train_index], qubits_measurements[test_index], \
            qubits_truths[train_index], qubits_truths[test_index]

        mv_grid = picklize("majority_vote_grid_search_cv_{fold}".format(fold=_i_fold)) \
            (majority_vote_grid_search_cv)(qubits_measurements_train, qubits_truths_train)
        print("Best params found by Grid Search: ")
        print(mv_grid.best_params_)
        print(pd.DataFrame(mv_grid.cv_results_)[
            ['param_hstgm__num_buckets', 'mean_test_score', 'std_test_score', 'rank_test_score']] \
                .sort_values('mean_test_score', ascending=False))

        curr_accuracy = classifier_test(mv_grid, qubits_measurements_train, qubits_measurements_test, 
                qubits_truths_train, qubits_truths_test)
        print("Train/test split {fold}-th fold accuracy: {accuracy}".format(fold=_i_fold, accuracy=curr_accuracy))
        clf_accuracies.append(curr_accuracy)
    
    avg_accuracy = sum(clf_accuracies) / len(clf_accuracies)
    print("Majority Vote with KFold Data Split: Average Accuracy = {accuracy}".format(accuracy=avg_accuracy))
    

def run_majority_vote_grid_search_cv_with_cross_validation_average():
    logging.info("Starting Voting Classifier Grid Search with Cross Validation Method.")

    qubits_measurements, qubits_truths = load_data()

    # construct iterator for training/testing dataset split
    # evenly distribute qubits with a certain number of photons captured
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    qubits_class = []
    assert(len(qubits_measurements) == len(qubits_truths))
    for index in range(len(qubits_measurements)):
        qubits_class.append(qubits_truths[index] * 100 + len(qubits_measurements[index]))
    cv_indices = kf.split(qubits_measurements, qubits_class)

    mv_grid = picklize('majority_vote_grid_search_cv_cv_average') \
        (majority_vote_grid_search_cv)(qubits_measurements, qubits_truths, cv=list(cv_indices))
    logging.info(mv_grid.cv_results_)

    best_accuracy = max(list(mv_grid.cv_results_['mean_test_score']))
    print("Best parameters found in Grid Search:")
    print(mv_grid.best_params_)
    print("Best average accuracy: {accuracy}".format(accuracy=best_accuracy))


def run_threshold_cutoff():
    """
    A toy data spliter is used primarily to show that a very high (99.975461%) but misleading accuracy
    can be achieved with bad training/testing split
    """
    logging.info("Starting Threshold Cutoff Classifier.")
    
    qubits_measurements, qubits_truths = load_data()
    qubits_measurements = np.array(qubits_measurements)
    qubits_truths = np.array(qubits_truths)
    
    # Data split
    kf = KFold(n_splits=5, shuffle=False)
    train_index, test_index = list(kf.split(qubits_measurements))[0]
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        qubits_measurements[train_index], qubits_measurements[test_index], qubits_truths[train_index], qubits_truths[test_index]

    tc_pipeline = Pipeline([
        ('hstgm', Histogramize(num_buckets=6, arrival_time_threshold=(0, POST_ARRIVAL_TIME_THRESHOLD))),
        ('clf', ThresholdCutoffClassifier())
    ])

    tc_pipeline = classifier_train(tc_pipeline, qubits_measurements_train, qubits_truths_train)
    classifier_test(tc_pipeline, qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test)


if __name__ == '__main__':
    # run_logistic_regression_with_kfold_data_split()
    # run_logistic_regression_grid_search_cv()
    # run_random_forest_with_kfold_data_split()
    # run_random_forest_grid_search_cv()
    # run_majority_vote_with_kfold_data_split()
    # run_majority_vote_grid_search_cv_with_cross_validation_average()
    run_threshold_cutoff()
    logging.info("Done.")
