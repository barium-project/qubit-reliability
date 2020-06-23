from src.features.build_features import *
from src.models.threshold_classifiers import *
from src.visualization.visualize import *

from sklearn.model_selection import  StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    X, y, s = load_data('ARTIFICIAL_V4', stats=True)
    print(s)
    qubits_class = [
        (s['file_range']['./data/artificial/v4/dark_tags_by_trial_with_decay_MC.csv'][0] <= i 
            and i <= s['file_range']['./data/artificial/v4/dark_tags_by_trial_with_decay_MC.csv'][1]) * 1000
        + y[i] * 100 
        + len(X[i]) for i in range(len(X))]
    indices = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X, qubits_class))
    pipeline = Pipeline([
        ("Histogramizer", Histogramizer(bins=11, range=(s['first_arrival'], s['last_arrival']))),
        ("Neural network", MLPClassifier(hidden_layer_sizes=(33, 33), activation='relu', solver='adam', max_iter=50, tol=0.001, verbose=True))]
    )
    for i in indices[:]:
        # Train neural network
        pipeline.fit(X[i[0]], y[i[0]])
        print(len(X[i[0]]))
        print(len(X[i[1]]))
        print()

        # Predict with threshold
        model = ThresholdCutoffClassifier(14)
        y_pred_threshold = model.predict(X[i[1]])
        dp_threshold = filter_datapoints(X[i[1]], y[i[1]], y_pred_threshold, indices=i[1])
        i_n = list(dp_threshold['i_n']) + list(dp_threshold['i_fn'])
        print(len(dp_threshold['i_n']))
        print(len(dp_threshold['i_fn']))
        print(len(dp_threshold['i_p']))
        print(len(dp_threshold['i_fp']))
        print((float(len(dp_threshold['i_p'])) + len(dp_threshold['i_n'])) / len(X[i[1]]))
        print()
        
        # Predict with neural network
        y_pred_neural_network_all = pipeline.predict(X[i[1]])
        dp_neural_network_all = filter_datapoints(X[i[1]], y[i[1]], y_pred_neural_network_all, indices=i[1])
        print(len(dp_neural_network_all['i_n']))
        print(len(dp_neural_network_all['i_fn']))
        print(len(dp_neural_network_all['i_p']))
        print(len(dp_neural_network_all['i_fp']))
        print((float(len(dp_neural_network_all['i_p'])) + len(dp_neural_network_all['i_n'])) / len(X[i[1]]))
        print()

        # Predict with the brights of the threshold using the neural network
        y_pred_neural_network = pipeline.predict(X[i_n])
        dp_neural_network = filter_datapoints(X[i_n], y[i_n], y_pred_neural_network, indices=i_n)
        print(len(dp_neural_network['i_n']))
        print(len(dp_neural_network['i_fn']))
        print(len(dp_neural_network['i_p']))
        print(len(dp_neural_network['i_fp']))

        print((float(len(dp_threshold['i_p'])) + len(dp_neural_network['i_n']) + len(dp_neural_network['i_p'])) / len(X[i[1]]))




