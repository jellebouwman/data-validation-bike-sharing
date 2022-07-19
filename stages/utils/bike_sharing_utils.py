import io
import typing
import zipfile
from datetime import datetime
from os.path import exists
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from alibi_detect.od import OutlierSeq2Seq


def load_bike_sharing_data(data_path: str, data_dir: Path) -> typing.Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame]:
    """Loads input data and does feature extraction. It downloads the data if it
     is present in the local file system.

    Args:
        data_path: Filepath with the data in the local file system
        data_dir: Path to the folder where to extract data if not present in the
         local file system. Please use the same folder as for 'data_path'

    Returns:
        Extracted features, extracted labels, original data
    """

    # Workaround before dvc remote storage is set up
    if not exists(data_path):
        content = requests.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            zip_ref.extractall(data_dir)

    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H")
    raw_data = pd.read_csv(data_path, header=0, sep=',',
                           parse_dates=[['dteday', 'hr']],
                           date_parser=custom_date_parser,
                           index_col='dteday_hr')

    raw_data['month'] = raw_data.index.map(lambda x: x.month)
    raw_data['hour'] = raw_data.index.map(lambda x: x.hour)
    raw_data['day'] = raw_data.index.map(lambda x: x.day_of_year)
    raw_data['weekday'] = raw_data.index.map(lambda x: x.weekday() + 1)

    raw_data['day_agg'] = raw_data['cnt'].resample('D').sum().resample(
        'H').ffill()
    raw_data['cnt_rel'] = raw_data['cnt'] / raw_data['day_agg']

    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hour',
                          'weekday']
    categorical_features = ['season', 'holiday', 'workingday']
    feature_names = numerical_features + categorical_features

    label = 'cnt'
    label_bin = 'cnt_bin'
    bin_size = 25
    raw_data[label_bin] = round(raw_data[label] / bin_size) * bin_size

    return raw_data[feature_names], raw_data[label], raw_data


def encode_decode_seq(X_mean: np.ndarray,
                      drift_detector: OutlierSeq2Seq) -> np.ndarray:
    """ Encodes and subsequently decodes the input sequence of data. This can be
     used to visually evaluate performance of the detector and whether the input
      data is outlier.

    Args:
        X_mean: Array of size 24 (one element for each hour in day). Each
        element contains information about rented bikes in the hour of the day.
        drift_detector: Trained outlier detector that should be used for
        encoding-decoding.

    Returns:
        Array of size 24 (one element for each hour in day). Each element
        contains reconstructed information about rented bikes in the hour of
        the day.

    """
    X_dec_mean = \
        drift_detector.seq2seq.decode_seq(X_mean[np.newaxis, :, np.newaxis])[0]
    X_dec_mean = X_dec_mean.reshape(X_dec_mean.shape[0], X_dec_mean.shape[1])
    X_dec_mean = X_dec_mean[0, :]
    return X_dec_mean


def get_working_day_aggregation(X_in: pd.DataFrame,
                                workingday: int = 0) -> pd.Series:
    """Calculates average relative ratio of bike rents for each hour in a day.

    Args:
        X_in: Dataframe with absolute value of rented bikes for each hour and
        day.
        workingday: 1 if the day is a working day and 0 if the day
        is holiday/weekend/...etc.

    Returns:
        Average relative ratio of bike rents for each hour in a (non) working
        day.
    """
    X = X_in.loc[X_in['workingday'] == workingday]
    X = pd.pivot_table(X, index=['day'], columns=['hour'], values=['cnt'])
    X = X.mean()
    X = X / X.sum()
    return X
