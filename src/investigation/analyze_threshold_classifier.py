from src.features.build_features import *
from src.models.threshold_classifiers import *
from src.visualization.visualize import *

from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    X, y = load_data()
    model = ThresholdCutoffClassifier(12)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred, digits=8))
    print(confusion_matrix(y, y_pred))

    X_p, y_p, X_n, y_n, X_fp, y_fp, X_fn, y_fn = filter_datapoints(X, y, y_pred)

    print(X_p[:30])
    visualize_photon_count(X_p)
    visualize_cumulative_arrival_distribution(X_p)
    visualize_individual_arrival_distribution(X_p[:30], limit=10, filename="X_p")

    print(X_n[:30])
    visualize_photon_count(X_n)
    visualize_cumulative_arrival_distribution(X_n)
    visualize_individual_arrival_distribution(X_n[:30], limit=10, filename="X_n")

    print(X_fp)
    visualize_photon_count(X_fp)
    visualize_cumulative_arrival_distribution(X_fp)
    visualize_individual_arrival_distribution(X_fp, limit=10, filename="X_fp")

    print(X_fn)
    visualize_photon_count(X_fn)
    visualize_cumulative_arrival_distribution(X_fn)
    visualize_individual_arrival_distribution(X_fn, limit=10, filename="X_fn")
