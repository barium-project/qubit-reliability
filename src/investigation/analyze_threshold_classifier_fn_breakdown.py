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
    i_fn_reg = []
    i_fn_decay = []
    for dp in dp['i_fn']:
        if s['file_range']['./data/artificial/v3/dark_tags_by_trial_no_decay_MC.csv'][0] <= dp and \
            dp <= s['file_range']['./data/artificial/v3/dark_tags_by_trial_no_decay_MC.csv'][1]:
            i_fn_reg.append(dp)
        if s['file_range']['./data/artificial/v3/dark_tags_by_trial_with_decay_MC.csv'][0] <= dp and \
            dp <= s['file_range']['./data/artificial/v3/dark_tags_by_trial_with_decay_MC.csv'][1]:
            i_fn_decay.append(dp)

    print('fn_reg: {}'.format(len(i_fn_reg)))
    print('fn_decay: {}'.format(len(i_fn_decay)))
