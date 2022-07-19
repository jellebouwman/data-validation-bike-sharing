import argparse
import os.path
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.bike_sharing_utils import get_working_day_aggregation, \
    load_bike_sharing_data
from utils.load_params import load_params


def get_data_split_by_month(X: pd.DataFrame, y: pd.Series) -> typing.Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the input features and labels to train and test datasets by month.

    Args:
        X: Dataframe of features
        y: Series of labels

    Returns:
        Train features, test features, train labels, test labels.
    """
    X_train = X.loc['2012-01-01 00:00:00':'2012-03-30 23:00:00']
    y_train = y.loc['2012-01-01 00:00:00':'2012-03-30 23:00:00']
    X_test = X.loc['2012-04-01 00:00:00':'2012-04-30 23:00:00']
    y_test = y.loc['2012-04-01 00:00:00':'2012-04-30 23:00:00']
    return X_train, X_test, y_train, y_test


def get_data_split_by_random(X: pd.DataFrame, y: pd.Series, test_size: float,
                             random_state: int) -> typing.Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the input features and labels to train and test datasets by random function

    Args:
        X: Dataframe of features
        y: Series of labels
        test_size: Size of test dataset.
        random_state: Random state used in train_test_split function

    Returns:
        Train features, test features, train labels, test labels.
    """
    X = X.loc['2012-01-01 00:00:00':'2012-04-30 23:00:00']
    y = y.loc['2012-01-01 00:00:00':'2012-04-30 23:00:00']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def save_data(data_dir: Path, X_train: pd.DataFrame, X_test: pd.DataFrame,
              y_train: pd.Series, y_test: pd.Series):
    """Saves the prepared train and test features and train and test labels to folder.

    Args:
        data_dir: Folderpath where to save data.
        X_train: Train features
        X_test:  Test features
        y_train: Train labels
        y_test:  Test labels
    """
    X_train.to_pickle(data_dir / 'X_train.pkl')
    X_test.to_pickle(data_dir / 'X_test.pkl')
    y_train.to_pickle(data_dir / 'y_train.pkl')
    y_test.to_pickle(data_dir / 'y_test.pkl')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_dir = Path(params['base']['data_dir'])
    data_fname = params['base']['data_fname']
    random_state = params['base']['random_state']
    test_size = params['data_preparation']['test_size']
    split_type = params['data_preparation']['data_split']

    data_path = os.path.join(data_dir, data_fname)
    X, y, raw_data = load_bike_sharing_data(data_path, data_dir)
    if split_type == 'month':
        X_train, X_test, y_train, y_test = get_data_split_by_month(X=X, y=y)
    else:
        X_train, X_test, y_train, y_test = get_data_split_by_random(X=X, y=y,
                                                                    test_size=test_size,
                                                                    random_state=random_state)
    save_data(data_dir, X_train, X_test, y_train, y_test)
    