#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

try:
    from pesq import pesq
    from pystoi import stoi
    import torchaudio
    import torchaudio.functional as F
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
except ImportError:
    raise ImportError(
        "Please install pesq, pystoi, torchaudio and retry: pip install stoi"
    )


def squim_metric(pred_x, gt_x, fs):

    if fs != 16000:
        gt_x = F.resample(gt_x, fs, 16000)
        pred_x = F.resample(pred_x, fs, 16000)

    subjective_model = SQUIM_SUBJECTIVE.get_model()
    objective_model = SQUIM_OBJECTIVE.get_model()

    mos = subjective_model(pred_x, gt_x)
    stoi, pesq, si_sdr = objective_model(pred_x)
    return {"mos": mos, "stoi": stoi, "pesq": pesq, "si_sdr": si_sdr}


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    scores = squim_metric(a, b, 16000)
    print(scores)
