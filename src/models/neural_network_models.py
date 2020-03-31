import logging

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.features.build_features import *
from src.util import *

# MLP Classifier
def mlp_grid_search_cv(qubits_measurements_train, qubits_truths_train, **kwargs):
    cv = kwargs['cv'] if 'cv' in kwargs else 4

    logging.info("Starting Grid Search with Cross Validation on MLP Classifier.")
    
    mlp_pipeline = Pipeline([
        # ('hstgm', Histogramize(num_buckets=6)),
        # ('hstgm', Histogramize(arrival_time_threshold=(0, POST_ARRIVAL_TIME_THRESHOLD))),
        ('hstgm', Histogramize(num_buckets=11, arrival_time_threshold=(PRE_ARRIVAL_TIME_THRESHOLD, POST_ARRIVAL_TIME_THRESHOLD))),
        ('clf', MLPClassifier(hidden_layer_sizes=(33, 33), activation='relu', solver='adam', random_state=RANDOM_SEED))
    ])

    mlp_param_grid = {
        # 'hstgm__num_buckets': range(1, 33),
        # 'hstgm__arrival_time_threshold': [(PRE_ARRIVAL_TIME_THRESHOLD, POST_ARRIVAL_TIME_THRESHOLD), (0, POST_ARRIVAL_TIME_THRESHOLD)],
        'clf__hidden_layer_sizes': [(33,) * n for n in range(2, 6)]
        # 'clf__learning_rate_init': [0.001, 0.0005],
        # 'clf__max_iter': [200, 500]
    }

    mlp_grid = GridSearchCV(mlp_pipeline, cv=cv, n_jobs=-1, param_grid=mlp_param_grid, scoring="accuracy", refit=True, verbose=2)
    mlp_grid.fit(qubits_measurements_train, qubits_truths_train)
    return mlp_grid

def run_mlp_classifier_in_paper():
    qubits_measurements, qubits_truths = load_data()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)

    logging.info("Histogramizing training and testing data.")
    histogramizer = Histogramize(num_buckets=6)
    qubits_measurements_train_histogram = histogramizer.transform(qubits_measurements_train)
    qubits_measurements_test_histogram = histogramizer.transform(qubits_measurements_test)

    mlp = picklize("mlp")(classifier_train)(
        MLPClassifier(hidden_layer_sizes=(8, 8), activation='relu', solver='adam'),  # 2-layer feed-forward neural network used in the paper
        qubits_measurements_train_histogram, qubits_truths_train)
    classifier_test(mlp, qubits_measurements_train_histogram, qubits_measurements_test_histogram, 
        qubits_truths_train, qubits_truths_test)

def run_mlp_grid_search_cv():
    qubits_measurements, qubits_truths = load_data()
    qubits_measurements_train, qubits_measurements_test, qubits_truths_train, qubits_truths_test = \
        train_test_split(qubits_measurements, qubits_truths, test_size=0.20, random_state=42)
        
    mlp_grid = picklize('mlp_grid_search_cv') \
        (mlp_grid_search_cv)(qubits_measurements_train, qubits_truths_train)
    logging.info(pd.DataFrame(mlp_grid.cv_results_))
    print("Best parameters found in Grid Search:")
    print(mlp_grid.best_params_)

    classifier_test(mlp_grid, qubits_measurements_train, qubits_measurements_test, 
        qubits_truths_train, qubits_truths_test)

def run_mlp_with_kfold_data_split():
    """
    Run the best model gotten from "run_mlp", but using 5-fold training/testing dataset split
    (32 neurons per layer, 2 layers, photons cutoff at BEST_ARRIVAL_TIME_THRESHOLD)
    This is the model presented on 01/30/2020 meeting
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

        mlp_pipeline = classifier_train(Pipeline([
                ('hstgm', Histogramize(num_buckets=6, arrival_time_threshold=(0, BEST_ARRIVAL_TIME_THRESHOLD))),
                ('clf', MLPClassifier(hidden_layer_sizes=(32, 32), activation='relu', solver='adam'))
            ]), qubits_measurements_train, qubits_truths_train)

        curr_accuracy = classifier_test(mlp_pipeline, qubits_measurements_train, qubits_measurements_test, 
                qubits_truths_train, qubits_truths_test)
        print("Train/test split {fold}-th fold accuracy: {accuracy}".format(fold=_i_fold, accuracy=curr_accuracy))
        clf_accuracies.append(curr_accuracy)
    
    avg_accuracy = sum(clf_accuracies) / len(clf_accuracies)
    print("MLP with KFold Data Split: Average Accuracy = {accuracy}".format(accuracy=avg_accuracy))

def run_mlp_with_cross_validation_average():
    """
    Run the best model gotten from "run_mlp" by cross validation method only.
    Training on 80% of the dataset and testing on 20% of the dataset 5 times, and take average of the accuracy
    """
    logging.info("Starting MLPClassifier testing with Cross Validation Method.")

    qubits_measurements, qubits_truths = load_data()

    mlp_pipeline = Pipeline([
            ('hstgm', Histogramize(num_buckets=11, arrival_time_threshold=(PRE_ARRIVAL_TIME_THRESHOLD, POST_ARRIVAL_TIME_THRESHOLD))),
            ('clf', MLPClassifier(hidden_layer_sizes=(33, 33), activation='relu', solver='adam', random_state=RANDOM_SEED))
        ])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    qubits_class = []
    assert(len(qubits_measurements) == len(qubits_truths))
    for index in range(len(qubits_measurements)):
        qubits_class.append(qubits_truths[index] * 100 + len(qubits_measurements[index]))
    cv_indices = kf.split(qubits_measurements, qubits_class)

    cv_scores = cross_validate(mlp_pipeline, qubits_measurements, qubits_truths, cv=list(cv_indices), scoring='accuracy', n_jobs=-1, verbose=2)
    print("Scores of Cross Validation Method on MLPClassifier: ")
    print(cv_scores)
    print("Average accuracy: {accuracy}".format(accuracy=
        sum(list(cv_scores['test_score'])) / len(list(cv_scores['test_score']))))

def run_mlp_grid_search_cv_with_cross_validation_average():
    """
    Run MLPClassifier with params Grid Search, 
    using the Cross Validation Method without splitting training/testing set beforehand
    """
    logging.info("Starting MLPClassifier Grid Search with Cross Validation Method.")

    qubits_measurements, qubits_truths = load_data()

    # construct iterator for training/testing dataset split
    # evenly distribute qubits with a certain number of photons captured
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    qubits_class = []
    assert(len(qubits_measurements) == len(qubits_truths))
    for index in range(len(qubits_measurements)):
        qubits_class.append(qubits_truths[index] * 100 + len(qubits_measurements[index]))
    cv_indices = kf.split(qubits_measurements, qubits_class)

    mlp_grid = picklize('mlp_grid_search_cv_cv_average') \
        (mlp_grid_search_cv)(qubits_measurements, qubits_truths, cv=list(cv_indices))
    logging.info(mlp_grid.cv_results_)

    best_accuracy = max(list(mlp_grid.cv_results_['mean_test_score']))
    print("Best parameters found in Grid Search:")
    print(mlp_grid.best_params_)
    print("Best average accuracy: {accuracy}".format(accuracy=best_accuracy))

def run_mlp_grid_search_cv_with_kfold_data_split(num_layers=2):
    """
    In each fold of the 5-fold training/testing data split, grid search for the best params in MLPClassifier
    and get the average accuracy
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

        # NOTE: 02/23 model, naming scheme to be improved
        mlp_grid = picklize('mlp_grid_search_cv_kfold_{layer}layers_0223_{fold}'.format(layer=num_layers, fold=_i_fold)) \
            (mlp_grid_search_cv)(qubits_measurements_train, qubits_truths_train, num_layers)
        print("Best params found by Grid Search: ")
        print(mlp_grid.best_params_)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1200):
        #     print(pd.DataFrame(mlp_grid.cv_results_)[
        #         ['param_hstgm__arrival_time_threshold', 'param_clf__hidden_layer_sizes', 'mean_test_score', 'std_test_score', 'rank_test_score']] \
        #             .sort_values('mean_test_score', ascending=False))

        curr_accuracy = classifier_test(mlp_grid, qubits_measurements_train, qubits_measurements_test, 
                qubits_truths_train, qubits_truths_test)
        print("Train/test split {fold}-th fold accuracy: {accuracy}".format(fold=_i_fold, accuracy=curr_accuracy))
        clf_accuracies.append(curr_accuracy)
    
    avg_accuracy = sum(clf_accuracies) / len(clf_accuracies)
    print("MLP with KFold Data Split: Average Accuracy = {accuracy}".format(accuracy=avg_accuracy))


if __name__ == '__main__':
    run_mlp_classifier_in_paper()
    # run_mlp_grid_search_cv()
    # run_mlp_with_kfold_data_split()
    # run_mlp_with_cross_validation_average()
    # run_mlp_grid_search_cv_with_cross_validation_average()
    # run_mlp_grid_search_cv_with_kfold_data_split(2) # broken since at least commit f99e61293fc30756266654fa0744ac6c494ecaff
