#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import logging

import os
import fnmatch
import librosa
import numpy as np
import soundfile as sf
import yaml
import kaldiio
from tqdm import tqdm
from typing import Dict, List


def find_files(
    root_dir: str, query: List[str] = ["*.flac", "*.wav"], include_root_dir: bool = True
) -> Dict[str, str]:
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (List[str]): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        Dict[str]: List of found filenames.

    """
    files = {}
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for q in query:
            for filename in fnmatch.filter(filenames, q):
                value = os.path.join(root, filename)
                if not include_root_dir:
                    value = value.replace(root_dir + "/", "")
                files[filename] = value
    return files


def audio_loader_setup(audio, io):
    # get ready compute embeddings
    if io == "kaldi":
        audio_files = kaldiio.load_scp(audio)
    elif io == "dir":
        audio_files = find_files(audio)
    elif io == "soundfile":
        audio_files = {}
        with open(audio) as f:
            for line in f.readlines():
                key, value = line.strip().split(maxsplit=1)
                if value.endswith("|"):
                    raise ValueError(
                        "Not supported wav.scp format. Set IO interface to kaldi"
                    )
                audio_files[key] = value
    return audio_files


def check_all_same(array):
    return np.all(array == array[0])


def wav_normalize(wave_array):
    if wave_array.ndim > 1:
        wave_array = wave_array[:, 0]
        logging.warning(
            "detect multi-channel data for mcd-f0 caluclation, use first channel"
        )
    if wave_array.dtype != np.int16:
        return np.ascontiguousarray(copy.deepcopy(wave_array.astype(np.float64)))
    # Convert the integer samples to floating-point numbers
    data_float = wave_array.astype(np.float64)

    # Normalize the floating-point numbers to the range [-1.0, 1.0]
    max_int16 = np.iinfo(np.int16).max
    normalized_data = data_float / max_int16
    return np.ascontiguousarray(copy.deepcopy(normalized_data))


def check_minimum_length(length, key_info):
    if "stoi" in key_info:
        # NOTE(jiatong): explicitly 0.256s as in https://github.com/mpariente/pystoi/pull/24
        if length < 0.3:
            return False
    if "pesq" in key_info:
        # NOTE(jiatong): check https://github.com/ludlows/PESQ/blob/master/pesq/cypesq.pyx#L37-L46
        if length < 0.25:
            return False
    if "visqol" in key_info:
        # NOTE(jiatong): check https://github.com/google/visqol/blob/master/src/image_patch_creator.cc#L50-L72
        if length < 1.0:
            return False
    if "sheet" in key_info:
        # NOTE(jiatong): check https://github.com/unilight/sheet/blob/main/hubconf.py#L13-L15
        if length < 0.065:
            return False
    return True


def load_score_modules(score_config, use_gt=True, use_gt_text=False, use_gpu=False):
    score_modules = {}
    for config in score_config:
        print(config, flush=True)
        if config["name"] == "mcd_f0":
            logging.info("Loading MCD & F0 evaluation...")
            from versa import mcd_f0

            score_modules["mcd_f0"] = {
                "module": mcd_f0,
                "args": {
                    "f0min": config.get("f0min", 0),
                    "f0max": config.get("f0max", 24000),
                    "mcep_shift": config.get("mcep_shift", 5),
                    "mcep_fftl": config.get("mcep_fftl", 1024),
                    "mcep_dim": config.get("mcep_dim", 39),
                    "mcep_alpha": config.get("mcep_alpha", 0.466),
                    "seq_mismatch_tolerance": config.get("seq_mismatch_tolerance", 0.1),
                    "power_threshold": config.get("power_threshold", -20),
                    "dtw": config.get("dtw", False),
                },
            }
            logging.info("Initiate MCD & F0 evaluation successfully.")

        elif config["name"] == "signal_metric":
            if not use_gt:
                logging.warning(
                    "Cannot use signal metric because no gt audio is provided"
                )
                continue

            logging.info("Loading signal metric evaluation...")
            from versa import signal_metric

            score_modules["signal_metric"] = {"module": signal_metric}
            logging.info("Initiate signal metric evaluation successfully.")

        elif config["name"] == "warpq":
            if not use_gt:
                logging.warning("Cannot use warpq because no gt audio is provided")
                continue

            logging.info("Loading WARPQ metric evaluation...")
            from versa import warpq, warpq_setup

            score_modules["warpq"] = {"module": warpq_setup()}
            logging.info("Initiate WARP-Q metric...")

        elif config["name"] == "discrete_speech":
            if not use_gt:
                logging.warning(
                    "Cannot use discrete speech metric because no gt audio is provided"
                )
                continue

            logging.info("Loading discrete speech evaluation...")
            from versa import discrete_speech_metric, discrete_speech_setup

            score_modules["discrete_speech"] = {
                "module": discrete_speech_metric,
                "args": {
                    "discrete_speech_predictors": discrete_speech_setup(use_gpu=use_gpu)
                },
            }
            logging.info("Initiate discrete speech evaluation successfully.")

        elif config["name"] == "pseudo_mos":
            logging.info("Loading pseudo MOS evaluation...")
            from versa import pseudo_mos_metric, pseudo_mos_setup

            predictor_dict, predictor_fs = pseudo_mos_setup(
                use_gpu=use_gpu,
                predictor_types=config.get("predictor_types", ["utmos"]),
                predictor_args=config.get("predictor_args", {}),
            )
            score_modules["pseudo_mos"] = {
                "module": pseudo_mos_metric,
                "args": {
                    "predictor_dict": predictor_dict,
                    "predictor_fs": predictor_fs,
                    "use_gpu": use_gpu,
                },
            }
            logging.info("Initiate pseudo MOS evaluation successfully.")

        elif config["name"] == "pesq":
            if not use_gt:
                logging.warning(
                    "Cannot use pesq metric because no gt audio is provided"
                )
                continue

            logging.info("Loadding pesq evaluation...")
            from versa import pesq_metric

            score_modules["pesq"] = {"module": pesq_metric}
            logging.info("Initiate pesq evaluation successfully.")

        elif config["name"] == "stoi":
            if not use_gt:
                logging.warning(
                    "Cannot use stoi metric because no gt audio is provided"
                )
                continue

            logging.info("Loading stoi evaluation...")
            from versa import stoi_metric

            score_modules["stoi"] = {"module": stoi_metric}
            logging.info("Initiate stoi evaluation successfully.")

        elif config["name"] == "visqol":
            if not use_gt:
                logging.warning(
                    "Cannot use visqol metric because no gt audio is provided"
                )
                continue

            logging.info("Loading visqol evaluation...")
            try:
                from versa import visqol_metric, visqol_setup
            except ImportError:
                logging.warning(
                    "VISQOL not installed, please check `tools` for installation guideline"
                )
                continue

            api, fs = visqol_setup(model=config.get("model", "default"))
            score_modules["visqol"] = {
                "module": visqol_metric,
                "args": {"api": api, "api_fs": fs},
            }
            logging.info("Initiate visqol evaluation successfully.")

        elif config["name"] == "speaker":
            if not use_gt:
                logging.warning(
                    "Cannot use speaker metric because no gt audio is provided"
                )
                continue

            logging.info("Loading speaker evaluation...")
            from versa import speaker_metric, speaker_model_setup

            spk_model = speaker_model_setup(
                model_tag=config.get("model_tag", "default"),
                model_path=config.get("model_path", None),
                model_config=config.get("model_config", None),
                use_gpu=use_gpu,
            )
            score_modules["speaker"] = {
                "module": speaker_metric,
                "args": {"model": spk_model},
            }
            logging.info("Initiate speaker evaluation successfully.")

        elif config["name"] == "sheet_ssqa":

            logging.info("Loading Sheet SSQA models for evaluation...")
            from versa import sheet_ssqa, sheet_ssqa_setup

            sheet_model = sheet_ssqa_setup(
                model_tag=config.get("model_tag", "default"),
                model_path=config.get("model_path", None),
                model_config=config.get("model_config", None),
                use_gpu=use_gpu,
            )
            score_modules["sheet_ssqa"] = {
                "module": sheet_ssqa,
                "args": {"model": sheet_model},
            }
            logging.info("Initiate Sheet SSQA evaluation successfully.")

        elif config["name"] == "squim_ref":
            if not use_gt:
                logging.warning("Cannot use squim_ref because no gt audio is provided")
                continue

            logging.info("Loadding squim metrics with reference")
            from versa import squim_metric

            score_modules["squim_ref"] = {
                "module": squim_metric,
            }
            logging.info("Initiate torch squim (with reference) successfully")

        elif config["name"] == "squim_no_ref":
            if not use_gt:
                logging.warning("Cannot use squim_ref because no gt audio is provided")
                continue

            logging.info("Loadding squim metrics with reference")
            from versa import squim_metric_no_ref

            score_modules["squim_no_ref"] = {
                "module": squim_metric_no_ref,
            }
            logging.info("Initiate torch squim (without reference) successfully")

        elif config["name"] == "espnet_wer":
            if not use_gt_text:
                logging.warning("Cannot use espnet_wer because no gt text is provided")
                continue
                # TODO(jiatong): add case for ground truth speech
                # (predict text for gt speech as well)

            logging.info("Loadding espnet_wer metric with reference text")
            from versa import espnet_levenshtein_metric, espnet_wer_setup

            score_modules["espnet_wer"] = {
                "module": espnet_levenshtein_metric,
                "args": espnet_wer_setup(
                    model_tag=config.get("model_tag", "default"),
                    beam_size=config.get("beam_size", 1),
                    text_cleaner=config.get("text_cleaner", "whisper_basic"),
                    use_gpu=use_gpu,
                ),
            }
            logging.info("Initiate ESPnet WER calculation successfully")

        elif config["name"] == "owsm_wer":
            if not use_gt_text:
                logging.warning("Cannot use owsm_wer because no gt text is provided")
                continue
                # TODO(jiatong): add case for ground truth speech
                # (predict text for gt speech as well)

            logging.info("Loadding owsm_wer metric with reference text")
            from versa import owsm_levenshtein_metric, owsm_wer_setup

            score_modules["owsm_wer"] = {
                "module": owsm_levenshtein_metric,
                "args": owsm_wer_setup(
                    model_tag=config.get("model_tag", "default"),
                    beam_size=config.get("beam_size", 1),
                    text_cleaner=config.get("text_cleaner", "whisper_basic"),
                    use_gpu=use_gpu,
                ),
            }
            logging.info("Initiate ESPnet-OWSM WER calculation successfully")

        elif config["name"] == "whisper_wer":
            if not use_gt_text:
                logging.warning("Cannot use whisper_wer because no gt text is provided")
                continue
                # TODO(jiatong): add case for ground truth speech
                # (predict text for gt speech as well)

            logging.info("Loadding whisper_wer metric with reference text")
            from versa import whisper_levenshtein_metric, whisper_wer_setup

            score_modules["whisper_wer"] = {
                "module": whisper_levenshtein_metric,
                "args": whisper_wer_setup(
                    model_tag=config.get("model_tag", "default"),
                    beam_size=config.get("beam_size", 1),
                    text_cleaner=config.get("text_cleaner", "whisper_basic"),
                    use_gpu=use_gpu,
                ),
            }
            logging.info("Initiate Whisper WER calculation successfully")
        
        elif config["name"] == "scoreq_ref":
            if not use_gt:
                logging.warning("Cannot use scoreq_ref because no gt audio is provided")
                continue

            logging.info("Loadding scoreq metrics with reference")
            from versa import scoreq_ref_setup, scoreq_ref
            model = scoreq_ref_setup(
                    data_domain=config.get("data_domain", "synthetic"),
                    cache_dir=config.get("model_cache", "./scoreq_pt-models"),
                    use_gpu=use_gpu,
                )

            score_modules["scoreq_ref"] = {
                "module": scoreq_ref,
                "model": model,
            }
            logging.info("Initiate scoreq (with reference) successfully")

        elif config["name"] == "scoreq_nr":
            logging.info("Loadding scoreq metrics without reference")
            from versa import scoreq_nr_setup, scoreq_nr
            model = scoreq_nr_setup(
                    data_domain=config.get("data_domain", "synthetic"),
                    cache_dir=config.get("model_cache", "./scoreq_pt-models"),
                    use_gpu=use_gpu,
                )

            score_modules["scoreq_nr"] = {
                "module": scoreq_nr,
                "model": model,
            }
            logging.info("Initiate scoreq (with reference) successfully")

    return score_modules


def use_score_modules(score_modules, gen_wav, gt_wav, gen_sr, text=None):
    utt_score = {}
    for key in score_modules.keys():
        if key == "mcd_f0":
            score = score_modules[key]["module"](
                gen_wav, gt_wav, gen_sr, **score_modules[key]["args"]
            )
        elif key == "signal_metric":
            score = score_modules[key]["module"](gen_wav, gt_wav)
        elif key == "discrete_speech":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["discrete_speech_predictors"],
                gen_wav,
                gt_wav,
                gen_sr,
            )
        elif key == "pseudo_mos":
            score = score_modules[key]["module"](
                gen_wav, gen_sr, **score_modules[key]["args"]
            )
        elif key == "pesq":
            score = score_modules[key]["module"](gen_wav, gt_wav, gen_sr)
        elif key == "stoi":
            score = score_modules[key]["module"](gen_wav, gt_wav, gen_sr)
        elif key == "visqol":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["api"],
                score_modules[key]["args"]["api_fs"],
                gen_wav,
                gt_wav,
                gen_sr,
            )
        elif key == "speaker":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["model"], gen_wav, gt_wav, gen_sr
            )
        elif key == "sheet_ssqa":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["model"], gen_wav, gen_sr
            )
        elif key == "squim_ref":
            score = score_modules[key]["module"](gen_wav, gt_wav, gen_sr)
        elif key == "squim_no_ref":
            score = score_modules[key]["module"](gen_wav, gen_sr)
        elif key == "espnet_wer" or key == "owsm_wer" or key == "whisper_wer":
            score = score_modules[key]["module"](
                score_modules[key]["args"],
                gen_wav,
                text,
                gen_sr,
            )
        elif key == "scoreq_ref":
            score = score_modules[key]["module"](
                score_modules[key]["model"],
                gen_wav, gt_wav, gen_sr)
        elif key == "scoreq_nr":
            score = score_modules[key]["module"](
                score_modules[key]["model"], 
                gen_wav, gen_sr) 
        else:
            raise NotImplementedError(f"Not supported {key}")

        logging.info(f"Score for {key} is {score}")
        utt_score.update(score)
    return utt_score


def list_scoring(
    gen_files,
    score_modules,
    gt_files=None,
    text_info=None,
    output_file=None,
    io="kaldi",
):
    if output_file is not None:
        f = open(output_file, "w", encoding="utf-8")

    score_info = []
    for key in tqdm(gen_files.keys()):
        if io == "kaldi":
            gen_sr, gen_wav = gen_files[key]
        elif io == "soundfile":
            gen_wav, gen_sr = sf.read(gen_files[key])
        else:
            raise NotImplementedError("Not supported io type: {}".format(io))
        gen_wav = wav_normalize(gen_wav)
        if not check_minimum_length(gen_wav.shape[0] / gen_sr, score_modules.keys()):
            logging.warning(
                "audio {} (generated, length {}) is too short to be evaluated with some metric metrics, skipping".format(
                    key, gen_wav.shape[0] / gen_sr
                )
            )
            continue

        if gt_files is not None:
            assert (
                key in gt_files.keys()
            ), "key {} not found in ground truth files".format(key)
        if gt_files is not None:
            if io == "kaldi":
                gt_sr, gt_wav = gt_files[key]
            else:
                gt_wav, gt_sr = sf.read(gt_files[key])
            gt_wav = wav_normalize(gt_wav)
            if check_all_same(gt_wav):
                logging.warning(
                    "skip audio with gt {}, as the gt audio has all the same value.".format(
                        key
                    )
                )
                continue
            if not check_minimum_length(gt_wav.shape[0] / gt_sr, score_modules.keys()):
                logging.warning(
                    "audio {} (ground truth, length {}) is too short to be evaluated with many metrics, skipping".format(
                        key, gt_wav.shape[0] / gt_sr
                    )
                )
                continue
        else:
            gt_wav = None
            gt_sr = None
        
        if text_info is not None:
            assert (
                key in text_info.keys()
            ), "key {} not found in ground truth transcription".format(key)
            text = text_info[key]
        else:
            text = None

        if gt_sr is not None and gen_sr > gt_sr:
            logging.warning(
                "Resampling the generated audio to match the ground truth audio"
            )
            gen_wav = librosa.resample(gen_wav, orig_sr=gen_sr, target_sr=gt_sr)
        elif gt_sr is not None and gen_sr < gt_sr:
            logging.warning(
                "Resampling the ground truth audio to match the generated audio"
            )
            gt_wav = librosa.resample(gt_wav, orig_sr=gt_sr, target_sr=gen_sr)

        utt_score = {"key": key}

        utt_score.update(use_score_modules(score_modules, gen_wav, gt_wav, gen_sr, text=text))
        del gen_wav
        del gt_wav

        if output_file is not None:
            f.write(f"{utt_score}\n")
        score_info.append(utt_score)
    logging.info("Scoring completed and save score at {}".format(output_file))
    return score_info


def load_summary(score_info):
    summary = {}
    for key in score_info[0].keys():
        if "ref_text" in key or "hyp_text" in key or key == "key":
            # NOTE(jiatong): skip text cases
            continue
        summary[key] = sum([score[key] for score in score_info])
        if "_wer" not in key and "_cer" not in key:
            # Average for non-WER/CER metrics
            summary[key] /= len(score_info)
    return summary


def load_corpus_modules(score_config, cache_forlder=".cache", use_gpu=False, io="kaldi"):
    score_modules = {}
    for config in score_config:
        if config["name"] == "fad":
            logging.info("Loading FAD evaluation with specific models...")
            from versa import fad_setup, fad_scoring
            fad_info = fad_setup(
                fad_embedding=config.get("model", "default"),
                baseline=config.get("baseline_audio", "default"),
                cache_dir=config.get("cache_dir"),
                use_inf=config.get("use_inf", False),
                io=io,
            )

            fad_key = "fad_{}".format(config.get("model", "default"))
 
            score_modules[fad_key] = {
                "module": fad_scoring,
                "args": fad_info,
            }
            logging.info("Initiate {} calculation evaluation successfully.".format(fad_key))
    
    return score_modules

def corpus_scoring(
        gen_files,
        score_modules,
        base_files=None,
        text_info=None,
        output_file=None,
        io="kaldi"
    ):
    pass