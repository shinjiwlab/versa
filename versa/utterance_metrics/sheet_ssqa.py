#!/usr/bin/env python3

# Copyright 2024 Wen-Chin Huang
# MIT License (https://opensource.org/licenses/MIT)
# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import librosa
import numpy as np
import torch


def sheet_ssqa_setup(
    model_tag="default", model_path=None, model_config=None, use_gpu=False
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if model_path is not None and model_config is not None:
        raise NotImplementedError(
            "Pending implementation for customized setup (Jiatong)"
        )
    else:
        if model_tag == "default":
            model_tag = "unilight/sheet:v0.1.0"
        model = torch.hub.load(
            "unilight/sheet:v0.1.0", "default", trust_repo=True, force_reload=True
        )

    model.model.to(device)
    return model


def sheet_ssqa(model, pred_x, fs):
    # NOTE(jiatong): current model only work for 16000 Hz
    if fs != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    return {"sheet_ssqa": model.predict(wav=torch.tensor(pred_x).float())}


if __name__ == "__main__":
    a = np.random.random(16000)
    model = sheet_ssqa_setup()
    print("metrics: {}".format(sheet_ssqa(model, a, 16000)))
