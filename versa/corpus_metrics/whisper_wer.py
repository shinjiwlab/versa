#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch
from Levenshtein import opcodes

try:
    import whisper
except ImportError:
    logging.warning(
        "Whisper is not properly installed. Please install following https://github.com/openai/whisper"
    )
    whisper = None

from espnet2.text.cleaner import TextCleaner

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def whisper_wer_setup(
    model_tag="default", beam_size=5, text_cleaner="whisper_basic", use_gpu=True
):
    if model_tag == "default":
        model_tag = "large"
    device = "cuda" if use_gpu else "cpu"
    if whisper is None:
        raise RuntimeError(
            "Whisper WER is used for evaluation while openai-whisper is not installed"
        )
    model = whisper.load_model(model_tag, device=device)
    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {"model": model, "cleaner": textcleaner, "beam_size": beam_size}
    return wer_utils


def whisper_levenshtein_metric(wer_utils, pred_x, ref_text, fs=16000):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: whisper model ("model"), text cleaner ("textcleaner"), and
            beam size ("beam size")
        pred_x (np.ndarray): test signal (time,)
        ref_text (string): reference transcript
        fs (int): sampling rate in Hz
    Returns:
        ret (dict): ditionary containing occurrences of edit operations
    """
    if fs != TARGET_FS:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS)
        fs = TARGET_FS
    with torch.no_grad():
        inf_txt = wer_utils["model"].transcribe(
            torch.tensor(pred_x).float(), beam_size=wer_utils["beam_size"]
        )["text"]

    ref_text = wer_utils["cleaner"](ref_text)
    pred_text = wer_utils["cleaner"](inf_txt)

    # process wer
    ref_words = ref_text.strip().split()
    pred_words = pred_text.strip().split()
    ret = {
        "whisper_hyp_text": pred_text,
        "ref_text": ref_text,
        "whisper_wer_delete": 0,
        "whisper_wer_insert": 0,
        "whisper_wer_replace": 0,
        "whisper_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["whisper_wer_" + op] = ret["whisper_wer_" + op] + inf_et - inf_st
        else:
            ret["whisper_wer_" + op] = ret["whisper_wer_" + op] + ref_et - ref_st
    total = (
        ret["whisper_wer_delete"]
        + ret["whisper_wer_replace"]
        + ret["whisper_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["whisper_wer_insert"]
        + ret["whisper_wer_replace"]
        + ret["whisper_wer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret.update(
        whisper_cer_delete=0,
        whisper_cer_insert=0,
        whisper_cer_replace=0,
        whisper_cer_equal=0,
    )
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["whisper_cer_" + op] = ret["whisper_cer_" + op] + inf_et - inf_st
        else:
            ret["whisper_cer_" + op] = ret["whisper_cer_" + op] + ref_et - ref_st
    total = (
        ret["whisper_cer_delete"]
        + ret["whisper_cer_replace"]
        + ret["whisper_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["whisper_cer_insert"]
        + ret["whisper_cer_replace"]
        + ret["whisper_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    return ret


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = whisper_wer_setup()
    print(
        "metrics: {}".format(
            whisper_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
