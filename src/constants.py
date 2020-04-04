RANDOM_SEED = 42

# Visualization constants
FIG_SIZE = (10.0, 8.0)

# Data constants
QUBIT_DATASET = {
    "V1": [
        ['./data/processed/v1/BrightTimeTagSet1.csv',
        './data/processed/v1/BrightTimeTagSet2.csv',
        './data/processed/v1/BrightTimeTagSet3.csv',
        './data/processed/v1/BrightTimeTagSet4.csv',
        './data/processed/v1/BrightTimeTagSet5.csv',],
        ['./data/processed/v1/DarkTimeTagSet1.csv',
        './data/processed/v1/DarkTimeTagSet2.csv',
        './data/processed/v1/DarkTimeTagSet3.csv',
        './data/processed/v1/DarkTimeTagSet4.csv',
        './data/processed/v1/DarkTimeTagSet5.csv',],
    ],
    "V3": [
        ['./data/processed/v3/bright_tags_by_trial.csv',],
        ['./data/processed/v3/dark_tags_by_trial.csv',],
    ],
    "V4": [
        ['./data/processed/v4/bright_tags_by_trial.csv',],
        ['./data/processed/v4/dark_tags_by_trial.csv',],
    ]
}
FIRST_ARRIVAL = 0.00541602
LAST_ARRIVAL = 0.009916040000000001
MAX_PHOTON_COUNT = 77


if __name__ == "__main__":
    pass
