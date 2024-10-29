#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch
from espnet2.bin.asr_inference import Speech2Text
from espnet2.text.cleaner import TextCleaner
from Levenshtein import opcodes

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def espnet_wer_setup(
    model_tag="default", beam_size=5, text_cleaner="whisper_basic", use_gpu=True
):
    if model_tag == "default":
        model_tag = "espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp"
    device = "cuda" if use_gpu else "cpu"
    model = Speech2Text.from_pretrained(
        model_tag=model_tag,
        device=device,
        beam_size=beam_size,
    )
    textcleaner = TextCleaner(text_cleaner)
    if "whisper" in text_cleaner:
        try:
            import whisper
        except ImportError:
            logging.warning(
                "Whipser-based cleaner is used but openai-whisper is not installed"
            )
    wer_utils = {"model": model, "cleaner": textcleaner, "beam_size": beam_size}
    return wer_utils


def espnet_predict(
    model,
    speech,
    fs: int,
    beam_size: int = 5,
):
    """Generate predictions using the espnet model. (from URGENT Challenge)

    Args:
        model (torch.nn.Module): espnet model.
        speech (np.ndarray): speech signal < 120s (time,)
        fs (int): sampling rate in Hz.
        beam_size (int): beam size used in beam search.
    Returns:
        text (str): predicted text
    """
    model.beam_search.beam_size = int(beam_size)

    assert fs == 16000, (fs, 16000)

    # assuming 10 tokens per second
    model.maxlenratio = -min(300, int((len(speech) / TARGET_FS) * 10))

    speech = librosa.util.fix_length(speech, size=(TARGET_FS * CHUNK_SIZE))
    text = model(speech)[0][0]

    return text


def espnet_levenshtein_metric(wer_utils, pred_x, ref_text, fs=16000):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: espnet model ("model"), text cleaner ("textcleaner"), and
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
        inf_txt = espnet_predict(
            wer_utils["model"],
            pred_x,
            fs,
            beam_size=wer_utils["beam_size"],
        )

    ref_text = wer_utils["cleaner"](ref_text)
    pred_text = wer_utils["cleaner"](inf_txt)

    # process wer
    ref_words = ref_text.split()
    pred_words = pred_text.split()
    ret = {
        "hyp_text": pred_text,
        "ref_text": ref_text,
        "espnet_wer_delete": 0,
        "espnet_wer_insert": 0,
        "espnet_wer_replace": 0,
        "espnet_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["espnet_wer_" + op] = ret["espnet_wer_" + op] + inf_et - inf_st
        else:
            ret["espnet_wer_" + op] = ret["espnet_wer_" + op] + ref_et - ref_st
    total = (
        ret["espnet_wer_delete"] + ret["espnet_wer_replace"] + ret["espnet_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["espnet_wer_insert"] + ret["espnet_wer_replace"] + ret["espnet_wer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret.update(
        espnet_cer_delete=0,
        espnet_cer_insert=0,
        espnet_cer_replace=0,
        espnet_cer_equal=0,
    )
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["espnet_cer_" + op] = ret["espnet_cer_" + op] + inf_et - inf_st
        else:
            ret["espnet_cer_" + op] = ret["espnet_cer_" + op] + ref_et - ref_st
    total = (
        ret["espnet_cer_delete"] + ret["espnet_cer_replace"] + ret["espnet_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["espnet_cer_insert"] + ret["espnet_cer_replace"] + ret["espnet_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    return ret


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = espnet_wer_setup()
    print(
        "metrics: {}".format(
            espnet_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
