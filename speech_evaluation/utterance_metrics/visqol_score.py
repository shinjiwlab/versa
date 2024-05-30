#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

import os

import librosa
import visqol
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2


def visqol_setup(model):
    # model name related to
    # https://github.com/google/visqol/tree/master/model

    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 48000

    if model == "default":
        model_tag = "libsvm_nu_svr_model.txt"
    elif model == "general":
        model_tag = "tcdaudio14_aacvopus_coresv_svrnsim_n.68_g.01_c1.model"
    elif model == "grid-search":
        model_tag = "tcdaudio_aacvopus_coresv_grid_nu0.3_c126.237786175_g0.204475514639.model"
    elif model == "speech":
        model_tag = "tcdvoip_nu.568_c5.31474325639_g3.17773760038_model.txt"
        config.options.use_speech_scoring = True
        config.audio.sample_rate = 16000
    else:
        raise NotImplementedError("Not a valid tag for model, check https://github.com/google/visqol/tree/master/model for details")


    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", model_tag)

    api = visqol_lib_py.VisqolApi()

    api.Create(config)

    return api, config.audio.sample_rate


def visqol_metric(api, api_fs, pred_x, gt_x, fs):
    if api_fs != fs:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=api_fs)
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=api_fs)
    

    similarity_result = api.Measure(gt_x, pred_x)

    return similarity_result.moslqo


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    predictor, fs = visqol_setup("default")
    print(visqol_metric(predictor, fs, a, b, 16000))
