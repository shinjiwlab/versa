#!/usr/bin/env python3

# Copyright 2024 Hye-jin Shim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch

try:
    from torch_log_wmse import LogWMSE

    logging.warning("Using the torch-log-wmse package for evaluation")
except ImportError:
    raise ImportError("Please install torch-log-wmse and retry")


def log_wmse(pred_x, gt_x, fs):
    audio_length = 1.0
    inst_log_wmse = LogWMSE(
        audio_length=audio_length, sample_Rate=fs, return_as_loss=True
    )
    return {"pesq": pesq_value}


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    scores = log_wmse(a, b, 16000)
    print(scores)
