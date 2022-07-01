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


def get_concept_drift_data(raw_data: pd.DataFrame, workingday: int) -> \
typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares training and test data for concept drift detector.

    Args:
        raw_data: Raw input data
        workingday: 1 if the day is a working day and 0 if the day is holiday/weekend/...etc.

    Returns:
        Train and test features for concept drift detector.
    """
    X = raw_data.loc[f'2011-01-01 00:00:00':f'2012-03-30 23:00:00'].loc[
        raw_data['workingday'] == workingday]
    X = pd.pivot_table(X, index=['day'], columns=['hour'],
                       values=['cnt_rel']).dropna()

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=5)
    X_train, X_test = X_train.to_numpy()[:, :, np.newaxis], X_test.to_numpy()[:,
                                                            :, np.newaxis]
    return X_train, X_test


def save_concept_drift_data(data_dir: Path, X_cd_wrk_train: np.ndarray,
                            X_cd_wrk_test: np.ndarray,
                            X_cd_nowrk_train: np.ndarray,
                            X_cd_nowrk_test: np.ndarray):
    """Saves training and test data for concept drift detector

    Args:
        data_dir: Folderpath where to save data.
        X_cd_wrk_train: Training data for working day.
        X_cd_wrk_test: Test data for working day.
        X_cd_nowrk_train: Training data for non-working day.
        X_cd_nowrk_test: Test data for non-working day.
    """
    np.save(data_dir / 'X_cd_wrk_train.npy', X_cd_wrk_train)
    np.save(data_dir / 'X_cd_wrk_test.npy', X_cd_wrk_test)
    np.save(data_dir / 'X_cd_nowrk_train.npy', X_cd_nowrk_train)
    np.save(data_dir / 'X_cd_nowrk_test.npy', X_cd_nowrk_test)


def save_concept_drift_data_for_demo(data_dir: Path, raw_data: pd.DataFrame):
    """Prepares and saves data for demonstration of concept drift. The concept
    drift is not present in the original data.
    It is created artificially in this function.

    Args:
        data_dir: Folderpath where to save data.
        raw_data: Raw input data.
    """
    # Simulate situation when there is a new competition that drops sales in
    # morning hours by 70%
    y_corrupted = raw_data.loc['2012-05-01 00:00:00':'2012-05-14 23:00:00']
    y_corrupted_ind = y_corrupted.loc[y_corrupted['hour'] > 12].index
    y_corrupted.loc[y_corrupted_ind, 'cnt'] = 0.3 * y_corrupted.loc[
        y_corrupted_ind, 'cnt']

    y_corrupted = get_working_day_aggregation(y_corrupted,
                                              workingday=0).droplevel(level=0)
    y_corrupted.to_pickle(data_dir / 'y_corrupted_demo.pkl')


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

    X_cd_wrk_train, X_cd_wrk_test = get_concept_drift_data(raw_data=raw_data,
                                                           workingday=1)
    X_cd_nowrk_train, X_cd_nowrk_test = get_concept_drift_data(
        raw_data=raw_data, workingday=0)
    save_concept_drift_data(data_dir=data_dir, X_cd_wrk_train=X_cd_wrk_train,
                            X_cd_wrk_test=X_cd_wrk_test,
                            X_cd_nowrk_train=X_cd_nowrk_train,
                            X_cd_nowrk_test=X_cd_nowrk_test)
    save_concept_drift_data_for_demo(data_dir=data_dir, raw_data=raw_data)
