#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
# Mainly adpated from ESPnet-SE (https://github.com/espnet/espnet.git)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import ci_sdr
import fast_bss_eval
from mir_eval.separation import bss_eval_sources


def calculate_si_snr(pred_x, gt_x, zero_mean=None, clamp_db=None, pairwise=False):
    # TODO(jiatong): pass zero_mean and clamp_db setup to the function
    pred_x = torch.from_numpy(pred_x).float()
    gt_x = torch.from_numpy(gt_x).float()

    si_snr_loss = fast_bss_eval.si_sdr_loss(
        est=pred_x,
        ref=gt_x,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        pairwise=pairwise,
    )
    return -float(si_snr_loss)


def calculate_ci_sdr(pred_x, gt_x, filter_length=512):
    # TODO(jiatong): pass filter_length to the function
    pred_x = torch.from_numpy(pred_x).float()
    gt_x = torch.from_numpy(gt_x).float()

    ci_sdr_loss = ci_sdr.pt.ci_sdr_loss(
        pred_x, gt_x, compute_permutation=False, filter_length=filter_length
    )
    return -float(ci_sdr_loss)


def signal_metric(pred_x, gt_x):
    # Expected input: (channel, samples)
    if pred_x.shape[1] != gt_x.shape[1]:
        min_audio_length = min(pred_x.shape[1], gt_x.shape[1])
        pred_x = pred_x[:, :min_audio_length]
        gt_x = pred_x[:, :min_audio_length]
    sdr, sir, sar, _ = bss_eval_sources(gt_x, pred_x, compute_permutation=False)
    si_snr = calculate_si_snr(pred_x, gt_x)
    ci_sdr = calculate_ci_sdr(pred_x, gt_x)
    return {
        "sdr": sdr[0],
        "sir": sir[0],
        "sar": sar[0],
        "si_snr": si_snr,
        "ci_sdr": ci_sdr,
    }


# debug code
if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    print(a, b)
    print("metrics: {}".format(signal_metric(a, b)))
