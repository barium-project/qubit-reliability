import numpy as np
from scipy import stats
from sklearn.model_selection import  KFold, StratifiedKFold, train_test_split

from src.features.build_features import *
from src.constants import *
from src.visualization.visualize import *


if __name__ == '__main__':
    X, y, s = load_data('ARTIFICIAL_V3_mini', stats=True)

    print('KFold without shuffle results in test set with only dark qubits')
    indices = list(KFold(n_splits=5, shuffle=False).split(X))
    # visualize_photon_count_group_by_traintest_truth(X, y, indices)
    for i in indices[:1]:
        print(stats.describe([len(x) for x in X[i[0]]]))
        print(stats.describe([len(x) for x in X[i[1]]]))
    print()
    
    print('KFold with shuffle')
    indices = list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X))
    # visualize_photon_count_group_by_traintest_truth(X, y, indices)
    # visualize_cumulative_arrival_distribution(*[X[i[0]] for i in indices], first_arrival=s['first_arrival'], last_arrival=s['last_arrival']) # Train
    # visualize_cumulative_arrival_distribution(*[X[i[1]] for i in indices], first_arrival=s['first_arrival'], last_arrival=s['last_arrival']) # Test
    # visualize_photon_count(*[X[i[0]] for i in indices], max_photon_count=s['max_count']) # Train
    # visualize_photon_count(*[X[i[1]] for i in indices], max_photon_count=s['max_count']) # Test
    for i in indices[:1]:
        print(stats.describe([len(x) for x in X[i[0]]]))
        print(stats.describe([len(x) for x in X[i[1]]]))
    print()
    
    print('StratifiedKFold on measurement class with shuffle')
    indices = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X, y))
    # visualize_photon_count_group_by_traintest_truth(X, y, indices)
    # visualize_cumulative_arrival_distribution(*[X[i[0]] for i in indices], first_arrival=s['first_arrival'], last_arrival=s['last_arrival']) # Train
    # visualize_cumulative_arrival_distribution(*[X[i[1]] for i in indices], first_arrival=s['first_arrival'], last_arrival=s['last_arrival']) # Test
    # visualize_photon_count(*[X[i[0]] for i in indices], max_photon_count=s['max_count']) # Train
    # visualize_photon_count(*[X[i[1]] for i in indices], max_photon_count=s['max_count']) # Test
    for i in indices[:1]:
        print(stats.describe([len(x) for x in X[i[0]]]))
        print(stats.describe([len(x) for x in X[i[1]]]))
    print()
    
    print('StratifiedKFold on measurement class and photon count with shuffle')
    qubits_class = [y[i] * 100 + len(X[i]) for i in range(len(X))] # y[i] * 100 separates bright from false; len(X[i]) separates photon counts
    indices = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X, qubits_class))
    # visualize_photon_count_group_by_traintest_truth(X, y, indices)
    # visualize_cumulative_arrival_distribution(*[X[i[0]] for i in indices], first_arrival=s['first_arrival'], last_arrival=s['last_arrival']) # Train
    # visualize_cumulative_arrival_distribution(*[X[i[1]] for i in indices], first_arrival=s['first_arrival'], last_arrival=s['last_arrival']) # Test
    # visualize_photon_count(*[X[i[0]] for i in indices], max_photon_count=s['max_count']) # Train
    # visualize_photon_count(*[X[i[1]] for i in indices], max_photon_count=s['max_count']) # Test
    for i in indices[:1]:
        print(stats.describe([len(x) for x in X[i[0]]]))
        print(stats.describe([len(x) for x in X[i[1]]]))
    print()
    