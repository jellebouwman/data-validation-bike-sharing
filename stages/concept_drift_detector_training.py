import argparse
from pathlib import Path

import numpy as np
from alibi_detect.od import OutlierSeq2Seq
from alibi_detect.utils.saving import save_detector

from utils.load_params import load_params


def train_concept_drift_detector(X: np.ndarray, detector_dir: Path,
                                 detector_fname: Path, latent_dim: int):
    """Trains and saves concept drift detector (=Seq2Seq detector)

    Args:
        X: Training data for the detector
        detector_dir: Folderpath where to save detector
        detector_fname: Filename of the saved detector
        latent_dim: Latent dimension of the encoder and decoder.
    """

    detector_dir.mkdir(exist_ok=True)

    seq_len = 24
    concept_drift_detector = OutlierSeq2Seq(1, seq_len,  # sequence length
                                            threshold=None,
                                            latent_dim=latent_dim)

    # train
    concept_drift_detector.fit(X, epochs=100, verbose=False)

    # Automatically infer threshold. Assume that 95% of training data are inliers
    concept_drift_detector.infer_threshold(X, threshold_perc=95)

    save_detector(concept_drift_detector, detector_dir / detector_fname)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_dir = Path(params['base']['data_dir'])
    detector_dir = Path(params['detector']['dir'])
    detector_fname_wrk = Path(
        params['detector']['concept_drift']['fname_working'])
    detector_fname_nowrk = Path(
        params['detector']['concept_drift']['fname_noworking'])
    latent_dim_wrk = params['detector']['concept_drift']['latent_dim_working']
    latent_dim_nowrk = params['detector']['concept_drift'][
        'latent_dim_noworking']

    X_wrk_train = np.load(data_dir / 'X_cd_wrk_train.npy')
    X_nowrk_train = np.load(data_dir / 'X_cd_nowrk_train.npy')

    train_concept_drift_detector(X=X_wrk_train, detector_dir=detector_dir,
                                 detector_fname=detector_fname_wrk,
                                 latent_dim=latent_dim_wrk)

    train_concept_drift_detector(X=X_nowrk_train, detector_dir=detector_dir,
                                 detector_fname=detector_fname_nowrk,
                                 latent_dim=latent_dim_nowrk)
