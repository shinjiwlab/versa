#!/usr/bin/env python3

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch

try:
    from nomad_versa import Nomad
except ImportError:
    logging.warning(
        "nomad is not installed. Please use `tools/install_nomad.sh` to install"
    )
    Nomad = None


def nomad_setup(use_gpu=False):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if Nomad is None:
        raise ModuleNotFoundError(
            "nomad is not installed. Please use `tools/install_nomad.sh` to install"
        )

    return Nomad(device=device)


def nomad(model, pred_x, gt_x, fs):
    # NOTE(hyejin): current model only have 16k options
    if fs != 16000:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    return {
        "nomad_score": model.predict("dir", test_path=pred_x, ref_path=gt_x),
    }


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    nomad_model = nomad_setup(use_gpu=True)
    fs = 16000
    nomad_score = nomad(nomad_model, a, b, fs)
    print(nomad_score)
