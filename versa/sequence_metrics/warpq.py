#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch

try:
    from WARPQ.WARPQmetric import warpqMetric
except ImportError:
    logger.info(
        "Please install WARP-Q from <versa_root>/tools/install_warpq.sh"
        "and retry after installation"
    )
    warpqMetric = None


def warpq_setup(
    fs=8000,
    n_mfcc=13,
    fmax=4000,
    patch_size=0.5,
    sigma=[[1, 0], [0, 3], [1, 3]],
    apply_vad=False,
):
    args = {
        "sr": fs,
        "n_mfcc": n_mfcc,
        "fmax": fmax,
        "patch_size": patch_size,
        "sigma": sigma,
        "apply_vad": apply_vad,
    }
    if warpqMetric is None:
        raise ImportError(
            "Please install WARP-Q from <versa_root>/tools/install_warpq.sh, and retry after installation"
        )
    model = warpqMetric(args)
    logger.info("Mapping model is not loaded for current implementation.")
    return model


def warpq(model, pred_x, gt_x, fs=8000):
    """
    Reference:
        W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines,
        “Speech quality assessmentwith WARP‐Q: From similarity to subsequence
        dynamic time warp cost,” IET Signal Processing, 16(9), 1050–1070 (2022)

    """
    target_fs = model.args["sr"]
    if target_fs != fs:
        gt_x = librosa.resample(gt_x, fs, target_fs)
        pred_x = librosa.resample(pred_x, fs, target_fs)

    score = model.evaluate_versa(gt_x, pred_x)
    return {"warpq": score}


if __name__ == "__main__":
    model = warpq_setup()
    test_audio = np.zeros(16000)
    ref_audio = np.zeros(16000)
    print(warpq(model, ref_audio, test_audio, 8000))
