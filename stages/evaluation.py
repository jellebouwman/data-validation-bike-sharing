import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alibi_detect.utils.saving import load_detector
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.bike_sharing_utils import encode_decode_seq
from utils.bike_sharing_utils import get_working_day_aggregation, \
    is_concept_drift
from utils.load_params import load_params


def generate_histogram_plots(plots_dir: Path, X_train: pd.DataFrame,
                             X_test: pd.DataFrame, is_drift: int, p_val: float):
    """Creates and saves histogram that compares value distributions of 'temp' and 'hum' features for training and
    test datasets. The result of detection from the covariate shift detector is printed into the plot.

    Args:
        plots_dir: Folderpath where to save plots.
        X_train: Training dataset.
        X_test: Test dataset.
        is_drift: Information whether it was detected drift between training and test datasets.
        p_val: P value of the drift detection
    """

    for col in ['temp', 'hum']:
        plt.figure()
        # weights = [np.ones_like(X_train[col]) / len(X_train), np.ones_like(X_test[col]) / len(X_test)]
        # plt.hist([X_train[col], X_test[col]], weights=weights, label=['train', 'test'])

        # plt.xlabel(col)
        # plt.ylabel('Relative ratio [-]')
        # plt.title(f"Value distribution of \'{col}\' for train and test datasets\n Is data drift: {is_drift}, p_val: {p_val}")
        # plt.legend(loc='upper right')
        # plt.savefig(plots_dir/f'{col}_feature_distribution.png')

        sns.histplot(data=X_train[col], stat='probability',
                     color="darkturquoise", bins=np.arange(0, 0.80, 0.05),
                     kde=True)
        sns.histplot(data=X_test[col], stat='probability', color="tomato",
                     bins=np.arange(0, 0.80, 0.05), kde=True)

        plt.grid(True)
        plt.xlabel(col)
        plt.ylabel('Relative ratio [-]')
        plt.title(
            f"Value distribution of \'{col}\' for train and test datasets\n Is data drift: {is_drift}, p_val: {p_val}")
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(plots_dir / f'{col}_feature_distribution.png')


def generate_working_day_json(plots_dir: Path, X_cd_wrk_test: np.ndarray,
                              X_cd_nowrk_test: np.ndarray,
                              X_cd_wrk_test_decoded: np.ndarray,
                              X_cd_nowrk_test_decoded: np.ndarray):
    """Creates and saves json files that can be later used by 'dvc plots' function. The function creates four plots:
    1) Average relative number of rented bikes in each hour of a working day
    2) Average relative number of rented bikes in each hour of a non-working day
    3) Encoded+decoded sequence of data from plot 1)
    4) Encoded+decoded sequence of data from plot 2)

    Args:
        plots_dir: Folderpath where to save json files.
        X_cd_wrk_test: Dataset with average relative number of rented bikes in each hour of a working day.
        X_cd_nowrk_test: Dataset with average relative number of rented bikes in each hour of a non-working day.
        X_cd_wrk_test_decoded: Encoded+decoded sequence of data for a working day.
        X_cd_nowrk_test_decoded: Encoded+decoded sequence of data for a non-working day.
    """
    with open(plots_dir / "cnt_average_wrk_day_orig.json", "w") as fd:
        json.dump({'cnt_average_wrk_day_orig': [{'rel_cnt': cnt, 'hour': ind}
                                                for ind, cnt in
                                                enumerate(X_cd_wrk_test)]}, fd,
                  indent=4)
    with open(plots_dir / "cnt_average_wrk_day_reconstructed.json", "w") as fd:
        json.dump({'cnt_average_wrk_day_reconstructed': [
            {'rel_cnt': cnt, 'hour': ind} for ind, cnt in
            enumerate(X_cd_wrk_test_decoded)]}, fd, indent=4)
    with open(plots_dir / "cnt_average_nowrk_day_orig.json", "w") as fd:
        json.dump({'cnt_average_nowrk_day_orig': [{'rel_cnt': cnt, 'hour': ind}
                                                  for ind, cnt in
                                                  enumerate(X_cd_nowrk_test)]},
                  fd, indent=4)
    with open(plots_dir / "cnt_average_nowrk_day_reconstructed.json",
              "w") as fd:
        json.dump({'cnt_average_nowrk_day_reconstructed': [
            {'rel_cnt': cnt, 'hour': ind} for ind, cnt in
            enumerate(X_cd_nowrk_test_decoded)]}, fd, indent=4)


def generate_working_day_plots(plots_dir: Path, X_cd_wrk_test: np.ndarray,
                               X_cd_nowrk_test: np.ndarray,
                               X_cd_wrk_test_decoded: np.ndarray,
                               X_cd_nowrk_test_decoded: np.ndarray):
    """Creates and saves plots showing concept drift detector performance during training. The function creates two plots:
    1) Training of concept drift detector for a working day.
    2) Training of concept drift detector for a non-working day.

    Args:
        plots_dir: Folderpath where to save json files.
        X_cd_wrk_test: Dataset with average relative number of rented bikes in each hour of a working day.
        X_cd_nowrk_test: Dataset with average relative number of rented bikes in each hour of a non-working day.
        X_cd_wrk_test_decoded: Encoded+decoded sequence of data for a working day.
        X_cd_nowrk_test_decoded: Encoded+decoded sequence of data for a non-working day.
    """

    # Inspect how well the sequence-to-sequence model can predict the signal by encoding and then decoding it
    plt.figure()
    ax = plt.gca()
    pd.DataFrame(X_cd_wrk_test).plot(ax=ax)
    pd.DataFrame(X_cd_wrk_test_decoded).plot(ax=ax)
    plt.title(
        f"Training of concept drift detector for working day\n Detector training error: {mean_absolute_error(X_cd_wrk_test_decoded, X_cd_wrk_test)}")
    plt.xlabel("Hour of day")
    plt.ylabel('Relative ratio of rented bikes')
    ax.legend(['Average working day pattern from data',
               'Reconstructed working day pattern from drift detector'])

    plt.savefig(plots_dir / f'detector_working_day.png')

    plt.figure()
    ax = plt.gca()
    pd.DataFrame(X_cd_nowrk_test_decoded).plot(ax=ax)
    pd.DataFrame(X_cd_nowrk_test).plot(ax=ax)
    plt.title(
        f"Training of concept drift detector for non working day\n Detector training error: {mean_absolute_error(X_cd_nowrk_test_decoded, X_cd_nowrk_test)}")
    plt.xlabel("Hour of day")
    plt.ylabel('Relative ratio of rented bikes')
    ax.legend(['Average non working day pattern from data',
               'Reconstructed non working day pattern from drift detector'])

    plt.savefig(plots_dir / f'detector_non_working_day.png')


def generate_concept_drift_plots(plots_dir: Path, X_wrk_test_pred: np.ndarray,
                                 X_nowrk_test_pred: np.ndarray,
                                 X_cd_wrk_test_decoded: np.ndarray,
                                 X_cd_nowrk_test_decoded: np.ndarray,
                                 concept_drift_report_wrk_day,
                                 concept_drift_report_nowrk_day):
    """Creates and saves plots showing how concept drift detector works on an ML model output. This can serve is another
    evaluation check during ML model and concept drift detector training. The function creates two plots:
    1) ML output and concept drift decoded sequence for a working day.
    2) ML output and concept drift decoded sequence for a non-working day.

    Args:
        plots_dir: Folderpath where to save json files.
        X_cd_wrk_test: Dataset with average relative number of rented bikes in each hour of a working day.
        X_cd_nowrk_test: Dataset with average relative number of rented bikes in each hour of a non-working day.
        X_cd_wrk_test_decoded: Encoded+decoded sequence of data for a working day.
        X_cd_nowrk_test_decoded: Encoded+decoded sequence of data for a non-working day.
    """

    plt.figure()
    ax = plt.gca()
    pd.DataFrame(X_wrk_test_pred).plot(ax=ax)
    pd.DataFrame(X_cd_wrk_test_decoded).plot(ax=ax)
    plt.xlabel("Hour of day")
    plt.ylabel('Relative ratio of rented bikes')
    plt.title(f"Comparison of working day pattern for ML model training\n\
               Concept drift detected: {is_concept_drift(concept_drift_report_wrk_day['data']['is_outlier'][0])}")
    ax.legend(['Prediction of ML for test data',
               'Reconstructed ML predictions from concept drift detector'])

    plt.savefig(plots_dir / f'concept_drift_working_day.png')

    plt.figure()
    ax = plt.gca()
    pd.DataFrame(X_nowrk_test_pred).plot(ax=ax)
    pd.DataFrame(X_cd_nowrk_test_decoded).plot(ax=ax)
    plt.xlabel("Hour of day")
    plt.ylabel('Relative ratio of rented bikes')
    plt.title(f"Comparison of non working day pattern for ML model training\n\
               Concept drift detected: {is_concept_drift(concept_drift_report_nowrk_day['data']['is_outlier'][0])}")
    ax.legend(['Prediction of ML for test data',
               'Reconstructed ML predictions from concept drift detector'])

    plt.savefig(plots_dir / f'concept_drift_non_working_day.png')


def evaluate_ml_model(data_dir: Path, model_dir: Path, model_fname: Path,
                      metrics_dir: Path):
    """Evaluate and saves metrics of trained ML model.

    Args:
        data_dir: Folderpath with test dataset.
        model_dir: Folderpath with the trained model.
        model_fname: Filename of the trained model.
        metrics_dir: Folderpath where to save metrics file.
    """

    X_test = pd.read_pickle(data_dir / 'X_test.pkl')
    y_test = pd.read_pickle(data_dir / 'y_test.pkl')

    clf = load(model_dir / model_fname)

    ml_metrics = {
        'mean_squared_error': mean_squared_error(y_test, clf.predict(X_test)), }

    json.dump(obj=ml_metrics, fp=open(metrics_dir / 'ml.json', 'w'), indent=4,
              sort_keys=True)


def evaluate_data_drift(data_dir: Path, detector_dir: Path,
                        data_drift_fname: Path, plots_dir: Path,
                        metrics_dir: Path):
    """Evaluates if there is a covariate shift in the input data. The function produces metrics file and plots.

    Args:
        data_dir: Folderpath with training and test datasets.
        detector_dir: Folderpath with trained covariate shift detector.
        data_drift_fname: Filename of the trained covariate shift detector.
        plots_dir: Folderpath where to save generated plots.
        metrics_dir: Folderpath where to save generated metrics.
    """

    X_test = pd.read_pickle(data_dir / 'X_test.pkl')
    X_train = pd.read_pickle(data_dir / 'X_train.pkl')

    data_drift_detector = load_detector(detector_dir / data_drift_fname)
    data_drift_report = data_drift_detector.predict(X_test.to_numpy())

    data_drift_metrics = {'is_drift': data_drift_report['data']['is_drift'],
        'drift_severity': float(
            data_drift_report['data']['distance'] / data_drift_report['data'][
                'distance_threshold']), }

    json.dump(obj=data_drift_metrics,
              fp=open(metrics_dir / 'data_drift.json', 'w'), indent=4,
              sort_keys=True)

    generate_histogram_plots(plots_dir=plots_dir, X_train=X_train,
                             X_test=X_test,
                             is_drift=data_drift_report['data']['is_drift'],
                             p_val=data_drift_report['data']['p_val'])


def evaluate_concept_drift(data_dir: Path, detector_dir: Path,
                           concept_drift_fname_wrk: Path,
                           concept_drift_fname_nowrk: Path, model_dir: Path,
                           model_fname: Path, plots_dir: Path,
                           metrics_dir: Path):
    """Evaluates if there is a concept drift in the input data. The function produces metrics file and plots.

    Args:
        data_dir: Folderpath with test dataset.
        detector_dir: Folderpath with trained concept drift detectors.
        concept_drift_fname_wrk: Filename of the trained concept drift detector for a working day.
        concept_drift_fname_nowrk: Filename of the trained concept drift detector for a non-working day.
        model_dir: Folderpath with the trained ML model.
        model_fname: Filename of the trained ML model.
        plots_dir: Folderpath where to save generated plots.
        metrics_dir: Folderpath where to save generated metrics.
    """

    X_test = pd.read_pickle(data_dir / 'X_test.pkl')

    clf = load(model_dir / model_fname)

    concept_drift_wrk = load_detector(detector_dir / concept_drift_fname_wrk)
    concept_drift_nowrk = load_detector(
        detector_dir / concept_drift_fname_nowrk)
    X_cd_wrk_test = np.load(data_dir / 'X_cd_wrk_test.npy').mean(axis=0)[:, 0]
    X_cd_nowrk_test = np.load(data_dir / 'X_cd_nowrk_test.npy').mean(axis=0)[:,
                      0]

    X_cd_wrk_test_decoded = encode_decode_seq(X_mean=X_cd_wrk_test,
                                              drift_detector=concept_drift_wrk)
    X_cd_nowrk_test_decoded = encode_decode_seq(X_mean=X_cd_nowrk_test,
                                                drift_detector=concept_drift_nowrk)

    # Detection of concept drift in ML output
    X_test_pred = X_test.copy()
    X_test_pred['cnt'] = clf.predict(X_test)
    X_test_pred['day'] = X_test_pred.index.map(lambda x: x.day_of_year)
    X_wrk_test_pred = get_working_day_aggregation(X_test_pred,
                                                  workingday=1).droplevel(0)
    X_wrk_test_pred_decoded = encode_decode_seq(X_mean=X_wrk_test_pred,
                                                drift_detector=concept_drift_wrk)
    X_nowrk_test_pred = get_working_day_aggregation(X_test_pred,
                                                    workingday=0).droplevel(0)
    X_nowrk_test_pred_decoded = encode_decode_seq(X_mean=X_nowrk_test_pred,
                                                  drift_detector=concept_drift_nowrk)

    concept_drift_report_wrk_day = concept_drift_wrk.predict(
        X_wrk_test_pred.to_numpy()[np.newaxis, :], outlier_type='feature')
    is_concept_drift_in_wrk_day = is_concept_drift(
        concept_drift_report_wrk_day['data']['is_outlier'][0])
    concept_drift_report_nowrk_day = concept_drift_nowrk.predict(
        X_nowrk_test_pred.to_numpy()[np.newaxis, :], outlier_type='feature')
    is_concept_drift_in_nowrk_day = is_concept_drift(
        concept_drift_report_nowrk_day['data']['is_outlier'][0])

    concept_drift_metrics = {
        'wrk_training_error': mean_absolute_error(X_cd_wrk_test_decoded,
                                                  X_cd_wrk_test),
        'nowrk_training_error': mean_absolute_error(X_cd_nowrk_test_decoded,
                                                    X_cd_nowrk_test),
        'is_drift_in_work_day': int(is_concept_drift_in_wrk_day),
        'is_drift_in_non_work_day': int(is_concept_drift_in_nowrk_day)}

    json.dump(obj=concept_drift_metrics,
              fp=open(metrics_dir / 'concept_drift.json', 'w'), indent=4,
              sort_keys=True)

    generate_working_day_plots(plots_dir=plots_dir, X_cd_wrk_test=X_cd_wrk_test,
                               X_cd_nowrk_test=X_cd_nowrk_test,
                               X_cd_wrk_test_decoded=X_cd_wrk_test_decoded,
                               X_cd_nowrk_test_decoded=X_cd_nowrk_test_decoded)
    generate_working_day_json(plots_dir=plots_dir, X_cd_wrk_test=X_cd_wrk_test,
                              X_cd_nowrk_test=X_cd_nowrk_test,
                              X_cd_wrk_test_decoded=X_cd_wrk_test_decoded,
                              X_cd_nowrk_test_decoded=X_cd_nowrk_test_decoded)
    generate_concept_drift_plots(plots_dir=plots_dir,
                                 X_wrk_test_pred=X_wrk_test_pred,
                                 X_nowrk_test_pred=X_nowrk_test_pred,
                                 X_cd_wrk_test_decoded=X_wrk_test_pred_decoded,
                                 X_cd_nowrk_test_decoded=X_nowrk_test_pred_decoded,
                                 concept_drift_report_wrk_day=concept_drift_report_wrk_day,
                                 concept_drift_report_nowrk_day=concept_drift_report_nowrk_day)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_dir = Path(params['base']['data_dir'])
    data_fname = params['base']['data_fname']
    detector_dir = Path(params['detector']['dir'])
    data_drift_fname = Path(params['detector']['data_drift']['fname'])
    concept_drift_fname_wrk = Path(
        params['detector']['concept_drift']['fname_working'])
    concept_drift_fname_nowrk = Path(
        params['detector']['concept_drift']['fname_noworking'])
    model_dir = Path(params['train']['model_dir'])
    model_fname = Path(params['train']['model_fname'])

    metrics_dir = Path(params['evaluation']['metrics_dir'])
    metrics_dir.mkdir(exist_ok=True)
    plots_dir = Path(params['evaluation']['plots_dir'])
    plots_dir.mkdir(exist_ok=True)

    evaluate_ml_model(data_dir=data_dir, model_dir=model_dir,
                      model_fname=model_fname, metrics_dir=metrics_dir)
    evaluate_data_drift(data_dir=data_dir, detector_dir=detector_dir,
                        data_drift_fname=data_drift_fname, plots_dir=plots_dir,
                        metrics_dir=metrics_dir)
    evaluate_concept_drift(data_dir=data_dir, detector_dir=detector_dir,
                           concept_drift_fname_wrk=concept_drift_fname_wrk,
                           concept_drift_fname_nowrk=concept_drift_fname_nowrk,
                           model_dir=model_dir, model_fname=model_fname,
                           plots_dir=plots_dir, metrics_dir=metrics_dir)