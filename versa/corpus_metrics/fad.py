#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch
import kaldiio
from tqdm import tqdm

from fadtk.fad_versa import FrechetAudioDistance
from fadtk.model_loader import get_all_models
from versa.scorer_shared import audio_loader_setup


def fad_setup(
        baseline,
        fad_embedding="default",
        cache_dir="versa_cache/fad",
        use_inf=True,
        io="kaldi",
    ):

    # get model
    models = {m.name: m for m in get_all_models()}
    if fad_embedding == "default":
        fad_embedding = "clap-laion-audio"
    model = models[fad_embedding]

    # setup fad object
    fad = FrechetAudioDistance(
        ml=model,
        load_model=True
    )

    return {
        "module": fad,
        "baseline": baseline,
        "cache_dir": cache_dir,
        "use_inf": use_inf,
        "io": io,
    }


def fad_scoring(pred_x, fad_info):

    cache_dir = fad_info["cache_dir"]

    # 1. Calculate embedding files for each dataset
    logging.info("[FAD] caching baseline embeddings...")
    baseline_files = audio_loader_setup(fad_info["baseline"], fad_info["io"])
    for key in tqdm(baseline_files.keys()):
        fad_info["module"].cache_embedding_file(baseline_files[key], cache_dir)
    logging.info("[FAD] Finished caching baseline embeddings.")

    logging.info("[FAD] caching eval embeddings...")
    eval_files = audio_loader_setup(pred_x, fad_info["io"])
    for key in tqdm(baseline_files.keys()):
        fad_info["module"].cache_embedding_file(eval_files[key], cache_dir)
    logging.info("[FAD] Finished caching eval embeddings.")

    if len(baseline_files) != len(eval_files):
        use_inf = True
    else:
        use_inf = fad_info["use_inf"]

    # 2. Calculate FAD
    if use_inf:
        score = fad_info["module"].score_inf(baseline_files, eval_files, cache_dir)
        return {"fad_overall": score.score, "fad_r2": score.r2}
    else:
        score = fad_info["module"].score(baseline_files, eval_files, cache_dir)
        return {"fad_overall": score}


if __name__ == "__main__":
    fad_info = fad_setup("test/test_samples/test1.scp")
    print(fad_scoring("test/test_samples/test2.scp", fad_info))
    

    
    
    

    
