#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import kaldiio
import librosa
import numpy as np
import torch
from fadtk.fad_versa import FrechetAudioDistance
from fadtk.model_loader import get_model
from tqdm import tqdm

from versa.scorer_shared import audio_loader_setup


def kid_setup(
    baseline,
    kid_embedding="default",
    cache_dir="versa_cache/kid",
    use_inf=True,
    io="kaldi",
):

    # get model
    model = get_model(kid_embedding)

    # setup kid object
    kid = FrechetAudioDistance(ml=model, load_model=True)

    return {
        "module": kid,
        "baseline": baseline,
        "cache_dir": cache_dir,
        "use_inf": use_inf,
        "io": io,
        "embedding": kid_embedding,
    }


def kid_scoring(pred_x, kid_info, key_info="kid"):

    cache_dir = kid_info["cache_dir"]

    # 1. Calculate embedding files for each dataset
    logging.info("[KID] caching baseline embeddings...")
    baseline_files = audio_loader_setup(kid_info["baseline"], kid_info["io"])
    for key in tqdm(baseline_files.keys()):
        kid_info["module"].cache_embedding_file(
            key, baseline_files[key], cache_dir + "/baseline"
        )
    logging.info("[KID] Finished caching baseline embeddings.")

    logging.info("[KID] caching eval embeddings...")
    eval_files = audio_loader_setup(pred_x, kid_info["io"])
    for key in tqdm(eval_files.keys()):
        kid_info["module"].cache_embedding_file(
            key, eval_files[key], cache_dir + "/eval"
        )
    logging.info("[KID] Finished caching eval embeddings.")


    # 2. Calculate kid
    score = kid_info["module"].score_kid(baseline_files, eval_files, cache_dir)
    # update score key with key_info
    updated_score = {key_info + key: value for key, value in score.items()}
    return updated_score


if __name__ == "__main__":
    kid_info = kid_setup("test/test_samples/test1.scp")
    print(kid_scoring("test/test_samples/test2.scp", kid_info))
