# qubit-reliability

## How to run
```
git clone https://github.com/barium-project/qubit-reliability.git   # Clone repo
cd qubit-reliability                                                # Move to directory
tar -xzvf data/processed/v3.zip -C data/processed                   # Unzip data
python3.6 -m venv env                                               # Create virtual environment
source env/bin/activate                                             # Activate virtual environmnet
pip install -r requirements.txt                                     # Install dependencies
python -m src.investigation.analyze_threshold_classifier            # Run analyze_threshold_classifier.py
```

## Directory Structure
```
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   │   └── v3         <- The fixed data.
│   │
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── investigation  <- Scripts to turn raw data into features for modeling
│   │   ├── analyze_splits.py
│   │   ├── analyze_threshold_classifier.py
│   │   └── analyze_neural_network_classifier.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── threshold_classifiers.py
│   │   ├── traditional_classifiers.py
│   │   └── neural_network_classifiers.py
│   │
│   └── visualization
│       └── visualize.py
│
├── README.md          <- The top-level README for developers using this project.
│
└── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                          generated with `pip freeze > requirements.txt`
```

## Conventions
Bright qubit = 0 = negative

Dark qubit = 1 = positive

Bright state error = Actually bright, but predicted dark = False positive

Dark state error = Actually dark, but predicted bright = False negative
