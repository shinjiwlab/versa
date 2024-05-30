#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

try:
    from pystoi import stoi
except ImportError:
    raise ImportError("Please install pystoi and retry: pip install stoi")


def stoi_metric(pred_x, gt_x, fs):
    score = stoi(gt_x, pred_x, fs, extended=False)
    return score

if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    scores = stoi_metric(a, b, 16000)
    print(scores)