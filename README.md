# VERSA
VERSA (Versatile Evaluation of Speech and Audio) is a toolkit dedicated to collecting evaluation metrics in speech and audio quality. Our goal is to provide a comprehensive connection to the cutting-edge techniques developed for evaluation. The toolkit is also tightly integrated into [ESPnet](https://github.com/espnet/espnet.git).

## Colab Demonstration

[Colab Demonstration at Interspeech2024 Tutorial](https://colab.research.google.com/drive/11c0vZxbSa8invMSfqM999tI3MnyAVsOp?usp=sharing)


## Install

The base-installation is as easy as follows:
```
git clone https://github.com/shinjiwlab/versa.git
cd versa
pip install .
```
or
```
pip install git+https://github.com/shinjiwlab/versa.git
```

As for collection purposes, VERSA instead of re-distributing the model, we try to align as much to the original API provided by the algorithm developer. Therefore, we have many dependencies. We try to include as many as default, but there are cases where the toolkit needs specific installation requirements. Please refer to our [list-of-metric section](https://github.com/shinjiwlab/versa?tab=readme-ov-file#list-of-metrics) for more details on whether the metrics are automatically included or not.  If not, we provide an installation guide or installers in `tools`.


## Quick test
```
python versa/test/test_script.py
```

## Usage

Simple usage case for a few samples.
```
# direct usage
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1 \
    --pred test/test_samples/test2 \
    --output_file test_result

# with scp-style input
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result

# with kaldi-ark style
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --io kaldi
  
# For text information
python versa/bin/scorer.py \
    --score_config egs/separate_metrics/wer.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --text test/test_samples/text
```

Use launcher with slurm job submissions
```
# use the launcher
# Option1: with gt speech
./launch.sh \
  <pred_speech_scp> \
  <gt_speech_scp> \
  <score_dir> \
  <split_job_num> 

# Option2: without gt speech
./launch.sh \
  <pred_speech_scp> \
  None \
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

We include [ ] and [x] to mark if the metirc is auto-installed in versa. 

|Number| Metric Name  (Auto-Install)  | Key in config | Key in report |  Code Source                                                                                                     | References                                                                                       |
|-|------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 | Mel Cepstral Distortion (MCD) [x]  | mcd_f0 | mcd | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel2/3220/9154/00407206.pdf) |
| 2 | F0 Correlation [x]  | mcd_f0 | f0_corr | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel7/9040208/9052899/09053512.pdf) |
| 3 | F0 Root Mean Square Error  [x] | mcd_f0 | f0_rmse | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel7/9040208/9052899/09053512.pdf) |
| 4 | Signal-to-interference  Ratio (SIR) [x]  | signal_metric | sir | [espnet](https://github.com/espnet/espnet) | - |
| 5 | Signal-to-artifact Ratio (SAR) [x]  | signal_metric | sar | [espnet](https://github.com/espnet/espnet) | - |
| 6 | Signal-to-distortion Ratio (SDR) [x]  | signal_metric | sdr | [espnet](https://github.com/espnet/espnet) | - |
| 7 | Convolutional scale-invariant signal-to-distortion ratio (CI-SDR) [x]  | signal_metric | ci-sdr | [ci_sdr](https://github.com/fgnt/ci_sdr) | [paper](https://arxiv.org/abs/2011.15003) |
| 8 | Scale-invariant signal-to-noise ratio (SI-SNR) [x]  | signal_metric | si-snr | [espnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/1711.00541) |
| 9 | Perceptual Evaluation of Speech Quality (PESQ) [x]  | pesq | pesq | [pesq](https://pypi.org/project/pesq/) | [paper](https://ieeexplore.ieee.org/document/941023) |
| 10 | Short-Time Objective Intelligibility (STOI) [x]  | stoi | stoi | [pystoi](https://github.com/mpariente/pystoi) | [paper](https://ieeexplore.ieee.org/document/5495701) |
| 11 | Speech BERT Score [x]  | discrete_speech | speech_bert | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 12 | Discrete Speech BLEU Score [x]  | discrete_speech | speech_belu | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 13 | Discrete Speech Token Edit Distance [x]  | discrete_speech | speech_token_distance | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 14 | UTokyo-SaruLab System for VoiceMOS Challenge 2022 (UTMOS) [x]  | pseudo_mos | utmos | [speechmos](https://github.com/tarepan/SpeechMOS) | [paper](https://arxiv.org/abs/2204.02152) |
| 15 | Deep Noise Suppression MOS Score of P.835 (DNSMOS) [x]  | pseudo_mos | dnsmos_overall | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2110.01763) |
| 16 | Deep Noise Suppression MOS Score of P.808 (DNSMOS) [x]  | pseudo_mos | dnsmos_p808 | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2005.08138) |
| 17 | Packet Loss Concealment-related MOS Score (PLCMOS) [x]  | pseudo_mos | plcmos | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2305.15127)|
| 18 | Virtual Speech Quality Objective Listener (VISQOL) [ ]  | visqol | visqol | [google-visqol](https://github.com/google/visqol) | [paper](https://arxiv.org/abs/2004.09584) |
| 19 | Speaker Embedding Similarity [x]  | speaker | spk_similarity | [espnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2401.17230) |
| 20 | PESQ in TorchAudio-Squim [x]  | squim_no_ref | torch_squim_pesq | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 21 | STOI in TorchAudio-Squim [x]  | squim_no_ref | torch_squim_stoi | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 22 | SI-SDR in TorchAudio-Squim [x]  | squim_no_ref | torch_squim_si_sdr | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 23 | MOS in TorchAudio-Squim [x]  | squim_ref | torch_squim_mos |[torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 24 | Singing voice MOS [x]  | singmos | singmos |[singmos](https://github.com/South-Twilight/SingMOS/tree/main) | [paper](https://arxiv.org/abs/2406.10911) |
| 25 | Log-Weighted Mean Square Error [x] | log_wmse | log_wmse |[log_wmse](https://github.com/nomonosound/log-wmse-audio-quality) |
| 26 | Dynamic Time Warping Cost Metric [ ] | warpq | warpq |[WARP-Q](https://github.com/wjassim/WARP-Q) | [paper](https://arxiv.org/abs/2102.10449) |
| 27 | Sheet SSQA MOS Models [x] | sheet_ssqa | sheet_ssqa |[Sheet](https://github.com/unilight/sheet/tree/main) | [paper](https://arxiv.org/abs/2411.03715) |
| 28 | ESPnet Speech Recognition-based Error Rate [x] | espnet_wer | espnet_wer |[ESPnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/pdf/1804.00015) |
| 29 | ESPnet-OWSM Speech Recognition-based Error Rate [x] | owsm_wer | owsm_wer |[ESPnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2309.13876) |
| 30 | OpenAI-Whisper Speech Recognition-based Error Rate [x] | whisper_wer | whisper_wer |[Whisper](https://github.com/openai/whisper) | [paper](https://arxiv.org/abs/2212.04356) |
| 31 | UTMOSv2: UTokyo-SaruLab MOS Prediction System [ ] | utmosv2 | utmosv2 |[UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | [paper](https://arxiv.org/abs/2409.09305) |
| 32 | Speech Contrastive Regression for Quality Assessment with reference (ScoreQ) [ ] |  scoreq_ref | scoreq_ref |[ScoreQ](https://github.com/ftshijt/scoreq/tree/main) | [paper](https://arxiv.org/pdf/2410.06675) |
| 33 | Speech Contrastive Regression for Quality Assessment without reference (ScoreQ) [ ] | scoreq_nr | scoreq_nr |[ScoreQ](https://github.com/ftshijt/scoreq/tree/main) | [paper](https://arxiv.org/pdf/2410.06675) |
| 34 | Emotion2vec similarity (emo2vec) [ ] | emo2vec_similarity | emotion_similarity | [emo2vec](https://github.com/ftshijt/emotion2vec/tree/main) | [paper](https://arxiv.org/abs/2312.15185) | 
| 35 | Speech enhancement-based SI-SNR [x] | se_snr | se_si_snr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 36 | Speech enhancement-based CI-SDR [x] | se_snr | se_ci_sdr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 37 | Speech enhancement-based SAR [x] | se_snr | se_sar | [ESPnet](https://github.com/espnet/espnet.git) | |
| 38 | Speech enhancement-based SDR [x] | se_snr | se_sdr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 39 | NOMAD: Unsupervised Learning of Perceptual Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality Assessment [ ] |  nomad | nomad |[Nomad](https://github.com/shimhz/nomad/tree/main) | [paper](https://arxiv.org/abs/2309.16284) |
| A few more in verifying/progresss|


## Contributor Guidelines

To implement a new metric to versa includes the following steps:

### Step1: Prepare metric
You may add the metric implementation in the following sub-directories (`versa/corpus_metrics`, `versa/utterance_metrics`, `versa/sequence_metrics`). Specifically,
- corpus_metrics: works for metrics that need the whole corpora to compute the metric (e.g., FAD or WER).
- utterance_metrics: works for utterance-level metrics
- sequence_metrics (will be deprecated in later versions and merged to utterance_metrics): stands for metrics that need comparing two feature sequences.

The typical format of the metric setup includes two functions, one for model setup, and the other for inference. Please refer to `versa/utterance/metrics/speaker.py` for an example of the implementation.

For special cases where the model setup is simple or not needed, we can simplify only one inference function without the setup function as exemplified in `versa/utterance_metrics/stoi.py`

**Special note**: 
- Please consider adding a simple test function at the end of the implementation.
- For consistency, we will have some fixed naming conventions to follow:
    - For the setup function, we will have an argument of `use_gpu` which is default set to `False`.
    - For the inference function, the previous preprocessor can provide five arguments so far (If you need more, please contact Jiatong Shi for further discussion on the interface):
        - model: the inference model to use
        - pred_x: audio signal to be evaluated
        - fs: audio signal's sampling rate
        - gt_x: [optional] the reference audio signal (automatically handled in the previous parts, the reference signal should also have the same sampling rate as the target signal to be evaluated)
        - ref_text: [optional] additional text information. It can be either the transcription for WER or the text description for audio signals.
- Toolkit development: to link the toolkit modeling to other implementations, the 
    - We recommend using the original tool/interface as much as possible if they can already be formatted into the current interface. However, if it is not, we recommend the following hacking options to link their methods to VERSA. This option also works for those packages that need very specific versions of the dependency packages :
        - 1. fork their repo
        - 2. add customized interface
        - 3. add localized install options to `tools`

### Step2: Register the metrics to the scoring list
For the second step, please add your metrics to the scoring list in `versa/scorer_shared.py`. Notably, you are expected to add the new metrics in both `load_score_modules()` and `use_score_modules()`.

At this step, please define a unique key for your metric to differentiate it from others. By referring to the key, you can declare the setup function in `load_score_modules()` and the inference function in `use_score_modules()`. Please refer to the existing examples so that they are following the same setup.

### Step3: Docs, Tests, Examples, and Code-wrapping up
At this step, the major implementation has been done and we mainly focus on the docs, test functions, examples, and code-wrapping up.

For Docs, please add your metrics to the `README.md` (List of Metrics Section). If the metrics need external tools from installers at `tools`, please include that with the `[ ]` mark in the first field (column).

For Tests, please add the local test functions at the corresponding metrics scripts temporarily (we will enable CI test in later stages).
- For metrics included in the default setup, you can add the test metric value in `test/test_general.py`
- For metrics not included in the default installation, you can add the test function in `test/test_{metric_name}.py`

For Examples, please put a separate `yaml` style configuration file in `egs/separate_metrics` following other examples.

For Code-wrapping up, we highly recommend you use `black` and `isort` to format your added scripts.
