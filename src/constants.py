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
    "ARTIFICIAL_V1": [ # No prep errors; default leak errors; bins=100
        ["./data/artificial/v1/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v1/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v1/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V2": [ # No prep errors; default leak errors; bins=100
        ["./data/artificial/v2/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v2/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v2/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V3": [ # No prep errors; 25% leak errors; bins=100
        ["./data/artificial/v3/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v3/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v3/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V3_mini": [ # No prep errors; 25% leak errors; bins=100; 0.1% of the data;
        ["./data/artificial/v3_mini/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v3_mini/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v3_mini/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V4": [ # No prep errors; 5% leak errors; bins=100
        ["./data/artificial/v4/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v4/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v4/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V5": [ # No prep errors; 25% leak errors; bins=20
        ["./data/artificial/v5/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v5/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v5/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V6": [ # No prep errors; 25% leak errors; bins=5
        ["./data/artificial/v6/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v6/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v6/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V7": [ # No prep errors; 25% leak errors; bins=400
        ["./data/artificial/v7/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v7/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v7/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V8": [ # No prep errors; default leak errors; bins=400
        ["./data/artificial/v8/bright_tags_by_trial_MC_0.csv"],
        ["./data/artificial/v8/dark_tags_by_trial_no_decay_MC_0.csv",
        "./data/artificial/v8/dark_tags_by_trial_with_decay_MC_0.csv"]
    ],
    "ARTIFICIAL_V8B": [ # No prep errors; default leak errors; bins=400; 4 billion data points
        ["./data/artificial/v8/bright_tags_by_trial_MC_0.csv",
        "./data/artificial/v8/bright_tags_by_trial_MC_1.csv"],
        ["./data/artificial/v8/dark_tags_by_trial_no_decay_MC_0.csv",
        "./data/artificial/v8/dark_tags_by_trial_no_decay_MC_1.csv",
        "./data/artificial/v8/dark_tags_by_trial_with_decay_MC_0.csv",
        "./data/artificial/v8/dark_tags_by_trial_with_decay_MC_1.csv"]
    ],
    "ARTIFICIAL_V9": [ # No prep errors; 25% leak errors; bins=60
        ["./data/artificial/v9/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v9/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v9/dark_tags_by_trial_with_decay_MC.csv"]
    ],
    "ARTIFICIAL_V10": [ # No prep errors; 25% leak errors; bins=80
        ["./data/artificial/v10/bright_tags_by_trial_MC.csv"],
        ["./data/artificial/v10/dark_tags_by_trial_no_decay_MC.csv",
        "./data/artificial/v10/dark_tags_by_trial_with_decay_MC.csv"]
    ],
}


if __name__ == "__main__":
    pass
