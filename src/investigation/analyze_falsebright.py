from src.features.build_features import *

if __name__ == "__main__":
    X, y, s = load_data('ARTIFICIAL_V7', stats=True)
    print(s)
    X_decay = X[s['file_range']['./data/artificial/v7/dark_tags_by_trial_with_decay_MC.csv'][0]:s['file_range']['./data/artificial/v7/dark_tags_by_trial_with_decay_MC.csv'][1] + 1]
    print(len(X_decay))

    a, b = 0, 0
    for x in X_decay:
        if len(x) > 14:
            a += 1
        else:
            b += 1
    print(a)
    print(b)