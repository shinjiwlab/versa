# speech evaluation
A toolkit dedicate for speech evaluation.


## Install
```
git clone https://github.com/ftshijt/speech_evaluation.git
cd speech_evaluation
pip install .
```
or
```
pip install git+https://github.com/ftshijt/speech_evaluation.git
```

You need to manually install visqol. We prepared a detailed installation guide at https://github.com/ftshijt/speech_evaluation/blob/main/speech_evaluation/utterance_metrics/INSTALL_visqol.md if you met issue with original repo https://github.com/google/visqol


## Quick test
```
python speech_evaluation/bin/test.py
```

## Usage

Simple usage case for a few samples.
```
# direct usage
python speech_evaluation/bin/scorer.py \
    --score_config egs/codec_16k.yaml \
    --gt egs/test/test1 \
    --output_file egs/test/test_result.txt
```

Use launcher with slurm job submissions
```
# use the launcher
# Option1: with gt speech
./launcher.sh \
  <pred_speech_scp> \
  <gt_speech_scp> \
  <score_dir> \
  <split_job_num> 

# Option2: without gt speech
./launcher.sh \
  <pred_speech_scp> None 
  <score_dir> \
  <split_job_num>

# aggregate the results
cat <score_dir>/result/*.result.cpu.txt > <score_dir>/utt_result.cpu.txt
cat <score_dir>/result/*.result.gpu.txt > <score_dir>/utt_result.gpu.txt

# show result
python scripts/show_result.py <score_dir>/utt_result.cpu.txt
python scripts/show_result.py <score_dir>/utt_result.gpu.txt 

```

Access `egs/*.yaml` for different config for differnt setups.

## List of Metrics
