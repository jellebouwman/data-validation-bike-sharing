import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from utils.load_params import load_params

from mlem.api import save


def train_model(data_dir: Path, model_fname: str,
                random_state: int, n_estimators: int):
    """Trains and saves ML model for prediction of rented bikes.

    Args:
        data_dir: Folderpath to the training data.
        model_dir: Folderpath where to save trained ML model.
        model_fname: Filename of the saved ML model.
        random_state: Random state for the Random Forest Classifier
        n_estimators: Number of estimators for the model.
    """
    X_train = pd.read_pickle(data_dir / 'X_train.pkl')
    y_train = pd.read_pickle(data_dir / 'y_train.pkl')

    clf = RandomForestClassifier(n_estimators=n_estimators,
        random_state=random_state)

    clf.fit(X_train, y_train)
    # model_dir.mkdir(exist_ok=True)
    # dump(clf, model_dir / model_fname)

    save(clf, model_fname, sample_data=X_train, description="Random Forest Classifier for bike sharing dataset", labels=['bike-sharing-model'])


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_dir = Path(params['base']['data_dir'])
    # model_dir = Path(params['train']['model_dir'])
    # model_fname = Path(params['train']['model_fname'])
    model_fname = params['train']['model_fname']
    random_state = params['base']['random_state']
    n_estimators = params['train']['n_estimators']

    train_model(data_dir=data_dir, model_fname=model_fname,
                random_state=random_state, n_estimators=n_estimators)
