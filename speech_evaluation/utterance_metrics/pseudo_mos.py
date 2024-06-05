#!/usr/bin/env python3

# Copyright 2023 Takaaki Saeki
# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import librosa
import numpy as np
import torch


def pseudo_mos_setup(predictor_types, predictor_args, use_gpu=False):
    # Supported predictor types: utmos, dnsmos, aecmos, plcmos
    # Predictor args: predictor specific args
    predictor_dict = {}
    predictor_fs = {}
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if (
        "aecmos" in predictor_types
        or "dnsmos" in predictor_types
        or "plcmos" in predictor_types
    ):
        try:
            import onnxruntime  # NOTE(jiatong): a requirement of aecmos but not in requirements
            import speechmos.aecmos as aecmos
            import speechmos.dnsmos as dnsmos
            import speechmos.plcmos as plcmos
        except ImportError:
            raise ImportError(
                "Please install speechmos for aecmos, dnsmos, and plcmos: pip install speechmos"
            )

    for predictor in predictor_types:
        if predictor == "utmos":
            utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong").to(
                device
            )
            predictor_dict["utmos"] = utmos
            predictor_fs["utmos"] = 16000
        elif predictor == "dnsmos":
            predictor_dict["dnsmos"] = dnsmos
            if "dnsmos" not in predictor_args:
                predictor_fs["dnsmos"] = 16000
            else:
                predictor_fs["dnsmos"] = predictor_args["dnsmos"]["fs"]
        elif predictor == "aecmos":
            predictor_dict["aecmos"] = aecmos
            if "aecmos" not in predictor_args:
                predictor_fs["aecmos"] = 16000
            else:
                predictor_fs["aecmos"] = predictor_args["aecmos"]["fs"]
        elif predictor == "plcmos":
            predictor_dict["plcmos"] = plcmos
            if "plcmos" not in predictor_args:
                predictor_fs["plcmos"] = 16000
            else:
                predictor_fs["plcmos"] = predictor_args["plcmos"]["fs"]
        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return predictor_dict, predictor_fs


def pseudo_mos_metric(pred, fs, predictor_dict, predictor_fs, use_gpu=False):
    scores = {}
    for predictor in predictor_dict.keys():
        if predictor == "utmos":
            if fs != predictor_fs["utmos"]:
                pred_utmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["utmos"]
                )
            else:
                pred_utmos = pred
            pred_tensor = torch.from_numpy(pred_utmos).unsqueeze(0)
            if use_gpu:
                pred_tensor = pred_tensor.to("cuda")
            score = predictor_dict["utmos"](pred_tensor, predictor_fs["utmos"])[
                0
            ].item()
            scores.update(utmos=score)

        elif predictor == "aecmos":
            if fs != predictor_fs["aecmos"]:
                pred_aecmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["aecmos"]
                )
            else:
                pred_aecmos = pred
            score = predictor["aecmos"].run(pred_aecmos, sr=fs)
            scores.update(
                aec_echo_mos=score["echo_mos"],
                aec_deg_mos=score["deg_mos"],
            )
        elif predictor == "dnsmos":
            if fs != predictor_fs["dnsmos"]:
                pred_dnsmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["dnsmos"]
                )
            else:
                pred_dnsmos = pred
            score = predictor["dnsmos"].run(pred_dnsmos, sr=fs)
            scores.update(dns_overall=score["ovrl_mos"], dns_p808=score["p808_mos"])
        elif predictor == "plcmos":
            if fs != predictor_fs["plcmos"]:
                pred_plcmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["plcmos"]
                )
            else:
                pred_plcmos = pred
            score = predictor["plcmos"].run(pred_plcmos, sr=fs)
            scores.update(plcmos=score["plcmos"])
        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return scores


if __name__ == "__main__":
    a = np.random.random(16000)
    print(a)
    predictor_dict, predictor_fs = pseudo_mos_setup(
        ["utmos", "aecmos", "dnsmos", "plcmos"],
        predictor_args={
            "aecmos": 16000,
            "dnsmos": 16000,
            "plcmos": 16000,
        },
    )
    scores = pseudo_mos_metric(
        a, fs=16000, predictor_dict=predictor_dict, predictor_fs=predictor_fs
    )
    print("metrics: {}".format())
