from versa.bin.scorer import find_files, load_score_modules, list_scoring, load_summary
import yaml
import os
import logging
import math

TEST_INFO = {'mcd': 3.944548129900429, 'f0rmse': 0.0021156428157651486, 'f0corr': 0.25196390785343026, 'sdr': 4.873952979593654, 'sir': float('inf'), 'sar': 4.873952979593654, 'si_snr': 1.0702767372131348, 'ci_sdr': 4.87394905090332, 'pesq': 1.5722705125808716, 'stoi': 0.007625108859647476, 'utmos': 1.9074420928955078, 'dns_overall': 1.452605403465056, 'dns_p808': 2.09430193901062, 'plcmos': 3.196744743982951, 'visqol': 1.714412514940845, 'spk_similarity': 0.8953613042831421}

def info_update():
    
    # find files
    if os.path.isdir("egs/test/test2"):
        gen_files = sorted(find_files("egs/test/test2"))

    # find reference file
    if os.path.isdir("egs/test/test1"):
        gt_files = sorted(find_files("egs/test/test1"))

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/codec_16k.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = list_scoring(
        gen_files, score_modules, gt_files, output_file=None
    )
    summary = load_summary(score_info)
    print("Summary: {}".format(load_summary(score_info)), flush=True)

    
    for key in summary:
        if math.isinf(TEST_INFO[key]) and math.isinf(summary[key]):
            # for sir"
            continue
        # the plc mos is undeterministic
        if abs(TEST_INFO[key] - summary[key]) > 1e-5 and key != "plcmos":
            raise ValueError("Value issue in the test case, might be some issue in scorer {}".format(key))
    print("check successful", flush=True)


if __name__ == "__main__":
    info_update()
