import argparse
import os.path
from pathlib import Path

import pandas as pd

from utils.bike_sharing_utils import load_bike_sharing_data, \
    get_working_day_aggregation
from utils.load_params import load_params


def save_concept_drift_data_for_demo(data_dir_demo: Path,
                                     raw_data: pd.DataFrame):
    """Prepares and saves data for demonstration of concept drift.
    The concept drift is not present in the original data.
    It is created artificially in this function.

    Args:
        data_dir_demo: Folderpath where to save data.
        raw_data: Raw input data.
    """
    # Simulate situation when there is a new competition that drops sales
    # in morning hours by 70%
    y_corrupted = raw_data.loc['2012-05-01 00:00:00':'2012-05-14 23:00:00']
    y_corrupted_ind = y_corrupted.loc[y_corrupted['hour'] > 12].index
    y_corrupted.loc[y_corrupted_ind, 'cnt'] = 0.3 * y_corrupted.loc[
        y_corrupted_ind, 'cnt']

    y_corrupted = get_working_day_aggregation(y_corrupted,
                                              workingday=0).droplevel(level=0)
    y_corrupted.to_pickle(data_dir_demo / 'y_corrupted_demo.pkl')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_dir = Path(params['base']['data_dir'])
    data_dir_demo = Path(params['base']['data_dir_demo'])
    data_fname = params['base']['data_fname']

    data_dir_demo.mkdir(exist_ok=True)
    data_path = os.path.join(data_dir, data_fname)
    X, y, raw_data = load_bike_sharing_data(data_path, data_dir)

    save_concept_drift_data_for_demo(data_dir_demo=data_dir_demo,
                                     raw_data=raw_data)
