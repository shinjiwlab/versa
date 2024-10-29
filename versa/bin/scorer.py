#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Scorer Interface for Speech Evaluation."""

import argparse
import fnmatch
import logging
import os
from typing import Dict, List

import kaldiio
import torch
import yaml

from versa.scorer_shared import list_scoring, load_score_modules, load_summary


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Speech Evaluation Interface")
    parser.add_argument(
        "--pred",
        type=str,
        help="Wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "--score_config", type=str, default=None, help="Configuration of Score Config"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Wav.scp for ground truth waveforms.",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Path of ground truth transcription."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path of directory to write the results.",
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=False, help="whether to use GPU if it can"
    )
    parser.add_argument(
        "--io",
        type=str,
        default="kaldi",
        choices=["kaldi", "soundfile"],
        help="io interface to use",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int,
        help="the overall rank in the batch processing, used to specify GPU rank",
    )
    return parser


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


def main():
    args = get_parser().parse_args()

    # In case of using `local` backend, all GPU will be visible to all process.
    if args.use_gpu:
        gpu_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_rank)
        logging.info(f"using device: cuda:{gpu_rank}")

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    if args.io == "kaldi":
        with open(args.pred) as f:
            gen_files = kaldiio.load_scp(args.pred)
    elif args.io == "dir":
        gen_files = find_files(args.pred)
    else:
        gen_files = {}
        with open(args.pred) as f:
            for line in f.readlines():
                key, value = line.strip().split(maxsplit=1)
                if value.endswith("|"):
                    raise ValueError(
                        "Not supported wav.scp format. Set IO interface to kaldi"
                    )
                gen_files[key] = value

    # find reference file
    if args.gt is not None:
        if args.io == "kaldi":
            gt_files = kaldiio.load_scp(args.gt)
        elif args.io == "dir":
            gt_files = find_files(args.gt)
        else:
            gt_files = {}
            with open(args.gt) as f:
                for line in f.readlines():
                    key, value = line.strip().split(maxsplit=1)
                    if value.endswith("|"):
                        raise ValueError("Not supported wav.scp format.")
                    gt_files[key] = value
    else:
        gt_files = None

    # fine ground truth transcription
    if args.text is not None:
        test_info = {}
        with open(args.text) as f:
            for line in f.readlines():
                key, value = line.strip().split(maxsplit=1)
                test_info[key] = value
    else:
        text_info = None

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if gt_files is not None and len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )

    logging.info("The number of utterances = %d" % len(gen_files))

    with open(args.score_config, "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gpu=args.use_gpu,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = list_scoring(
        gen_files,
        score_modules,
        gt_files,
        text_info,
        output_file=args.output_file,
        io=args.io,
    )
    logging.info("Summary: {}".format(load_summary(score_info)))


if __name__ == "__main__":
    main()
