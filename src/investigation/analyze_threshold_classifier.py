from src.features.build_features import *
from src.models.threshold_classifiers import *
from src.visualization.visualize import *

from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    X, y, s = load_data('ARTIFICIAL_V3', stats=True)
    print(s)
    model = ThresholdCutoffClassifier(14)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred, digits=8))
    print(confusion_matrix(y, y_pred))

    dp = filter_datapoints(X, y, y_pred, indices=list(range(0, len(X))))

    print(dp['X_p'][:30])
    visualize_photon_count(dp['X_p'], max_photon_count=s['max_count'])
    visualize_cumulative_arrival_distribution(dp['X_p'], first_arrival=s['first_arrival'], last_arrival=s['last_arrival'])
    visualize_individual_arrival_distribution(dp['X_p'], limit=10, first_arrival=s['first_arrival'], last_arrival=s['last_arrival'])

    print(dp['X_n'][:30])
    visualize_photon_count(dp['X_n'], max_photon_count=s['max_count'])
    visualize_cumulative_arrival_distribution(dp['X_n'], first_arrival=s['first_arrival'], last_arrival=s['last_arrival'])
    visualize_individual_arrival_distribution(dp['X_n'], limit=10, first_arrival=s['first_arrival'], last_arrival=s['last_arrival'])

    print(dp['X_fp'])
    visualize_photon_count(dp['X_f'], max_photon_count=s['max_count'])
    visualize_cumulative_arrival_distribution(dp['X_fp'], first_arrival=s['first_arrival'], last_arrival=s['last_arrival'])
    visualize_individual_arrival_distribution(dp['X_fp'], limit=10, first_arrival=s['first_arrival'], last_arrival=s['last_arrival'])

    print(dp['X_fn'])
    visualize_photon_count(dp['X_f'], max_photon_count=s['max_count'])
    visualize_cumulative_arrival_distribution(dp['X_fn'], first_arrival=s['first_arrival'], last_arrival=s['last_arrival'])
    visualize_individual_arrival_distribution(dp['X_fn'], limit=10, first_arrival=s['first_arrival'], last_arrival=s['last_arrival'])
