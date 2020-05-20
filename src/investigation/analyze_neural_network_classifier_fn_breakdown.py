from src.features.build_features import *
from src.visualization.visualize import *

from sklearn.model_selection import  StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import csv

if __name__ == "__main__":
    X, y, s = load_data('ARTIFICIAL_V8', stats=True)
    print(s)
    qubits_class = [y[i] * 100 + len(X[i]) for i in range(len(X))]
    indices = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X, qubits_class))
    pipeline = Pipeline([
        ("Histogramizer", Histogramizer(bins=11, range=(s['first_arrival'], s['last_arrival']))),
        ("Neural network", MLPClassifier(hidden_layer_sizes=(33, 33), activation='relu', solver='adam', max_iter=50, tol=0.001, verbose=True))]
    )
    for i in indices[:]:
        pipeline.fit(X[i[0]], y[i[0]])
        y_pred = pipeline.predict(X[i[1]])

        print(classification_report(y[i[1]], y_pred, digits=8))
        print(confusion_matrix(y[i[1]], y_pred))
        dp = filter_datapoints(X[i[1]], y[i[1]], y_pred, indices=i[1])

        i_fn_reg = []
        i_fn_decay = []
        for dp in dp['i_fn']:
            if (s['file_range']['./data/artificial/v8/dark_tags_by_trial_no_decay_MC_0.csv'][0] <= dp and \
                dp <= s['file_range']['./data/artificial/v8/dark_tags_by_trial_no_decay_MC_0.csv'][1]):
                i_fn_reg.append(dp)
            if (s['file_range']['./data/artificial/v8/dark_tags_by_trial_with_decay_MC_0.csv'][0] <= dp and \
                dp <= s['file_range']['./data/artificial/v8/dark_tags_by_trial_with_decay_MC_0.csv'][1]):
                i_fn_decay.append(dp)

        print('fn_reg: {}'.format(len(i_fn_reg)))
        print('fn_decay: {}'.format(len(i_fn_decay)))
