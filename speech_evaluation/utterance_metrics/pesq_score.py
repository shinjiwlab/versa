#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import librosa

import numpy as np
from pesq import pesq
try:
    from pesq import pesq
    logging.warning("Using the PESQ package for evaluation")
except ImportError:
    raise ImportError("Please install pesq and retry: pip install pesq")


def pesq_metric(pred_x, gt_x, fs):
    if fs == 8000:
        pseq = pesq(8000, gt_x, pred_x, 'nb')
    elif fs < 16000:
        logging.info("not support fs {}, resample to 8khz".format(fs))
        new_gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=8000)
        new_pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=8000)
        pseq = pesq(16000, new_gt_x, new_pred_x, 'nb')
    elif fs == 16000:
        pseq = pesq(16000, gt_x, pred_x, 'wb')
    else:
        logging.info("not support fs {}, resample to 16khz".format(fs))
        new_gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        new_pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
        pseq = pesq(16000, new_gt_x, new_pred_x, 'wb')
    
    return pseq

if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    scores = pesq_metric(a, b, 16000)
    print(scores)
