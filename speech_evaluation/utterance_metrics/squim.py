#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch

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

    gt_x = torch.from_numpy(gt_x).unsqueeze(0)
    gt_x = gt_x.float()

    pred_x = torch.from_numpy(pred_x).unsqueeze(0)
    pred_x = pred_x.float()

    subjective_model = SQUIM_SUBJECTIVE.get_model()
    mos = subjective_model(pred_x, gt_x)

    #objective_model = SQUIM_OBJECTIVE.get_model()
    #stoi, pesq, si_sdr = objective_model(pred_x)
    
    return {"mos": mos.detach().numpy()}


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    scores = squim_metric(a, b, 16000)
    print(scores)
