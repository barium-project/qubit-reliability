RANDOM_SEED = 42

# Visualization constants
FIG_SIZE = (10.0, 8.0)

# Data constants
QUBIT_DATASET = {
    "REAL_V1": [ # Original
        ["./data/processed/v1/BrightTimeTagSet1.csv",
        "./data/processed/v1/BrightTimeTagSet2.csv",
        "./data/processed/v1/BrightTimeTagSet3.csv",
        "./data/processed/v1/BrightTimeTagSet4.csv",
        "./data/processed/v1/BrightTimeTagSet5.csv",],
        ["./data/processed/v1/DarkTimeTagSet1.csv",
        "./data/processed/v1/DarkTimeTagSet2.csv",
        "./data/processed/v1/DarkTimeTagSet3.csv",
        "./data/processed/v1/DarkTimeTagSet4.csv",
        "./data/processed/v1/DarkTimeTagSet5.csv",],
    ],
    "REAL_V2": [ # Contains fix for shifting time
        ["./data/processed/v2/bright_tags_by_trial.csv",],
        ["./data/processed/v2/dark_tags_by_trial.csv",],
    ],
    "ARTIFICIAL_V1": [ # With prep errors and leak errors
        ["./data/artificial/v1/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v1/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v1/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V2": [ # With no prep errors but with leak errors
        ["./data/artificial/v2/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v2/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v2/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V3": [ # With no prep errors and 25% leak errors
        ["./data/artificial/v3/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v3/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v3/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V3_mini": [ # With no prep errors and 25% leak errors; 0.1% of the data
        ["./data/artificial/v3_mini/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v3_mini/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v3_mini/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V4": [ # With no prep errors and 5% leak errors
        ["./data/artificial/v4/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v4/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v4/dark_tags_by_trial_with_decay_MC.csv"]
    ]
}


if __name__ == "__main__":
    pass
