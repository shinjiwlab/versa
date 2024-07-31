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

| Metric Name      | Key in config | Key in report | Details | Code Source                                                                                                     | References                                                                                       |
|------------------|---------------|---------------|---------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Mel Cepstral Distortion (MCD) | mcd_f0 | mcd | | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | https://ieeexplore.ieee.org/iel2/3220/9154/00407206.pdf |
| F0 Correlation | mcd_f0 | f0_corr | | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | https://ieeexplore.ieee.org/iel7/9040208/9052899/09053512.pdf |
| F0 Root Mean Square Error | mcd_f0 | f0_rmse | | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | https://ieeexplore.ieee.org/iel7/9040208/9052899/09053512.pdf |
| Signal-to-infererence Ratio (SIR) | signal_metric | sir | | [espnet](https://github.com/espnet/espnet) | - |
| Signal-to-artifact Ratio (SAR) | signal_metric | sar | | [espnet](https://github.com/espnet/espnet) | - |
| Signal-to-distortion Ratio (SDR) | signal_metric | sdr | | [espnet](https://github.com/espnet/espnet) | - |
| Convolutional scale-invariant signal-to-distortion ratio (CI-SDR) | signal_metric | ci-sdr | | [ci_sdr](https://github.com/fgnt/ci_sdr) | https://arxiv.org/abs/2011.15003 |
| Scale-invariant signal-to-noise ratio (SI-SNR) | signal_metric | si-snr | | [espnet](https://github.com/espnet/espnet) | https://arxiv.org/abs/1711.00541 |
| Perceptual Evaluation of Speech Quality (PESQ) | pesq | pesq | | [pesq](https://pypi.org/project/pesq/) | https://ieeexplore.ieee.org/document/941023 |
| Short-Time Objective Intelligibility (STOI) | stoi | stoi | | [pystoi](https://github.com/mpariente/pystoi) | https://ieeexplore.ieee.org/document/5495701 |
| Speech BERT Score | discrete_speech | speech_bert | | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | https://arxiv.org/abs/2401.16812 |
| Discrete Speech BLEU Score | discrete_speech | speech_belu | | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | https://arxiv.org/abs/2401.16812 |
| Discrete Speech Token Edit Distance | discrete_speech | speech_token_distance | | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | https://arxiv.org/abs/2401.16812 |
| UTokyo-SaruLab System for VoiceMOS Challenge 2022 (UTMOS) | pseudo_mos | utmos | | [speechmos](https://github.com/tarepan/SpeechMOS) | https://arxiv.org/abs/2204.02152 |
| Deep Noise Suppression MOS Score of P.835 (DNSMOS) | pseudo_mos | dnsmos_overall | | [speechmos (MS)](https://pypi.org/project/speechmos/) | https://arxiv.org/abs/2110.01763 |
| Deep Noise Suppression MOS Score of P.808 (DNSMOS) | pseudo_mos | dnsmos_p808 | | [speechmos (MS)](https://pypi.org/project/speechmos/) | https://arxiv.org/abs/2005.08138 |
| Packet Loss Concealment-related MOS Score (PLCMOS) | pseudo_mos | plcmos | | [speechmos (MS)](https://pypi.org/project/speechmos/) | https://arxiv.org/abs/2305.15127|
| Virtual Speech Quality Objective Listener (VISQOL) | visqol | visqol | | [google-visqol](https://github.com/google/visqol) | https://arxiv.org/abs/2004.09584 |
| Speaker Embedding Similarity | speaker | spk_similarity | | [espnet](https://github.com/espnet/espnet) | https://arxiv.org/abs/2401.17230 |
| PESQ in TorchAudio-Squim | squim | torch_squim_pesq | | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | https://arxiv.org/abs/2304.01448 |
| STOI in TorchAudio-Squim | squim | torch_squim_stoi | | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | https://arxiv.org/abs/2304.01448 |
| SI-SDR in TorchAudio-Squim | squim | torch_squim_si_sdr | | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | https://arxiv.org/abs/2304.01448 |
| MOS in TorchAudio-Squim | squim | torch_squim_mos |  | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | https://arxiv.org/abs/2304.01448 |
