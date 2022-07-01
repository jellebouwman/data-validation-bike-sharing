import argparse
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alibi_detect.utils.saving import load_detector

from utils.bike_sharing_utils import encode_decode_seq, is_concept_drift
from utils.load_params import load_params


def plot_detector_feature_score(y_corrupted: pd.Series, cd_report: TypedDict,
                                y_corrupted_decoded: np.ndarray,
                                cd_threshold: float, plots_dir: Path):
    """This function produces plot showing feature score (~reconstruction error)
    of concept drift detector for each hour of a day.

    Args:
        y_corrupted: Data that has been artificially adjusted.
        cd_report: Report created by the concept drift detector.
        y_corrupted_decoded: Decoded data from concept drift detector.
        cd_threshold: Threshold of the concept drift detector.
        plots_dir: Folderpath where to save plots.
    """

    y_df = pd.DataFrame(y_corrupted, columns=['original'])
    y_df['decoded'] = y_corrupted_decoded
    y_df['feature_score'] = cd_report['data']['feature_score'][0]
    y_df['reconstruction_error'] = np.abs(cd_report['data']['feature_score'][0])
    y_df['feature_score_zoom'] = 10 * y_df['feature_score']

    plt.figure()
    ax = plt.gca()

    sns.lineplot(data=24 * [cd_threshold], color='red', ax=ax)
    sns.barplot(data=y_df.reset_index(), x='hour', y='feature_score_zoom',
                ax=ax, alpha=1)

    ax.lines[0].set_linestyle("--")
    plt.ylabel('Feature score')
    plt.xlabel("Hour of day")
    plt.grid(True)
    plt.title(
        f"Detector's feature score (= reconstruction error)\nfor a non-working day with concept drift")
    ax.legend([f'Threshold of detector\n={round(cd_threshold, 5)}'])

    plt.savefig(plots_dir / f'demo_feature_score.png')


def plot_non_working_day_pattern(y_corrupted: pd.Series,
                                 y_corrupted_decoded: np.ndarray,
                                 concept_drift_in_data: bool, plots_dir: Path):
    """This function produces plot comparing input and output from the trained concept drift detector.

    Args:
        y_corrupted: Data that has been artificially adjusted.
        y_corrupted_decoded: Decoded data from concept drift detector.
        concept_drift_in_data: Boolean value whether there is detected concept drift in the data.
        plots_dir: Folderpath where to save plots.
    """

    plt.figure()
    ax = plt.gca()
    sns.lineplot(data=y_corrupted, ax=ax)
    sns.lineplot(data=y_corrupted_decoded, ax=ax)

    plt.grid(True)
    plt.xlabel("Hour of day")
    plt.ylabel('Relative ratio of rented bikes')
    plt.title(f"Non-working day pattern for data with present concept drift\n\
                Concept drift detected: {concept_drift_in_data}")
    ax.legend(['Input data to concept drift detector',
               'Reconstructed data from concept drift detector'])

    plt.savefig(plots_dir / f'corrupted_demo.png')


def evaluate_corrupted_data(data_dir: Path, detector_dir: Path,
                            concept_drift_fname_nowrk: Path):
    """Evaluates what happens when there is a concept drift in the data.
    Note that the concept drift is not present in the original data,
    but is artificially created.

    Args:
        data_dir: Folderpath with the data containing the concept drift.
        detector_dir: Folderpath with the trained concept drift detector.
        concept_drift_fname_nowrk: Filename of the trained concept drift detector.
    """

    y_corrupted = pd.read_pickle(data_dir / 'y_corrupted_demo.pkl')

    concept_drift_nowrk = load_detector(
        detector_dir / concept_drift_fname_nowrk)

    cd_report = concept_drift_nowrk.predict(
        y_corrupted.to_numpy()[np.newaxis, :], outlier_type='feature')
    concept_drift_in_data = is_concept_drift(cd_report['data']['is_outlier'][0])
    y_corrupted_decoded = encode_decode_seq(X_mean=y_corrupted,
                                            drift_detector=concept_drift_nowrk)

    # plt.figure()
    # ax = plt.gca()
    # pd.DataFrame(y_corrupted).plot(ax=ax)
    # pd.DataFrame(y_corrupted_decoded).plot(ax=ax)

    plot_non_working_day_pattern(y_corrupted=y_corrupted,
                                 y_corrupted_decoded=y_corrupted_decoded,
                                 concept_drift_in_data=concept_drift_in_data,
                                 plots_dir=plots_dir)

    plot_detector_feature_score(y_corrupted=y_corrupted, cd_report=cd_report,
                                y_corrupted_decoded=y_corrupted_decoded,
                                cd_threshold=concept_drift_nowrk.threshold,
                                plots_dir=plots_dir)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_dir = Path(params['base']['data_dir'])
    detector_dir = Path('../', params['detector']['dir'])
    concept_drift_fname_nowrk = Path(
        params['detector']['concept_drift']['fname_noworking'])

    plots_dir = Path(params['evaluation']['plots_dir'])
    plots_dir.mkdir(exist_ok=True)

    evaluate_corrupted_data(data_dir=data_dir, detector_dir=detector_dir,
                            concept_drift_fname_nowrk=concept_drift_fname_nowrk)
