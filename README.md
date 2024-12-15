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

As for collection purposes, VERSA instead of re-distributing the model, we try to align as much to the original API provided by the algorithm developer. Therefore, we have many dependencies. We try to include as many as default, but there are cases where the toolkit needs specific installation requirements. Please refer to our [list-of-metric section](https://github.com/shinjiwlab/versa?tab=readme-ov-file#list-of-metrics) for more details on whether the metrics are automatically included or not.  If not, we provide an installation guide or installers in `tools`.


## Quick test
```
python versa/test/test_general.py

# test metrics with additional installation
python versa/test/test_{metric}.py
```

## Usage

Simple usage case for a few samples.
```
# direct usage
python versa/bin/scorer.py \
    --score_config egs/general.yaml \
    --gt test/test_samples/test1 \
    --pred test/test_samples/test2 \
    --output_file test_result

# with scp-style input
python versa/bin/scorer.py \
    --score_config egs/general.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result

# with kaldi-ark style
python versa/bin/scorer.py \
    --score_config egs/general.yaml \
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

Access `egs/*.yaml` for different configs for different setups.

## List of Metrics

We include x mark if the metric is auto-installed in versa. 

|Number| Auto-Install | Metric Name  (Auto-Install)  | Key in config | Key in report |  Code Source                                                                                                     | References                                                                                       |
|---|---|------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 | x | Mel Cepstral Distortion (MCD)  | mcd_f0 | mcd | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel2/3220/9154/00407206.pdf) |
| 2 | x | F0 Correlation | mcd_f0 | f0_corr | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel7/9040208/9052899/09053512.pdf) |
| 3 | x | F0 Root Mean Square Error  | mcd_f0 | f0_rmse | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel7/9040208/9052899/09053512.pdf) |
| 4 | x | Signal-to-interference  Ratio (SIR)  | signal_metric | sir | [espnet](https://github.com/espnet/espnet) | - |
| 5 | x | Signal-to-artifact Ratio (SAR)  | signal_metric | sar | [espnet](https://github.com/espnet/espnet) | - |
| 6 | x | Signal-to-distortion Ratio (SDR)  | signal_metric | sdr | [espnet](https://github.com/espnet/espnet) | - |
| 7 | x | Convolutional scale-invariant signal-to-distortion ratio (CI-SDR)  | signal_metric | ci-sdr | [ci_sdr](https://github.com/fgnt/ci_sdr) | [paper](https://arxiv.org/abs/2011.15003) |
| 8 | x | Scale-invariant signal-to-noise ratio (SI-SNR)  | signal_metric | si-snr | [espnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/1711.00541) |
| 9 | x | Perceptual Evaluation of Speech Quality (PESQ)  | pesq | pesq | [pesq](https://pypi.org/project/pesq/) | [paper](https://ieeexplore.ieee.org/document/941023) |
| 10 | x | Short-Time Objective Intelligibility (STOI)  | stoi | stoi | [pystoi](https://github.com/mpariente/pystoi) | [paper](https://ieeexplore.ieee.org/document/5495701) |
| 11 | x | Speech BERT Score  | discrete_speech | speech_bert | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 12 | x | Discrete Speech BLEU Score  | discrete_speech | speech_belu | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 13 | x | Discrete Speech Token Edit Distance  | discrete_speech | speech_token_distance | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 14 | x | UTokyo-SaruLab System for VoiceMOS Challenge 2022 (UTMOS)  | pseudo_mos | utmos | [speechmos](https://github.com/tarepan/SpeechMOS) | [paper](https://arxiv.org/abs/2204.02152) |
| 15 | x | Deep Noise Suppression MOS Score of P.835 (DNSMOS)  | pseudo_mos | dnsmos_overall | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2110.01763) |
| 16 | x | Deep Noise Suppression MOS Score of P.808 (DNSMOS)  | pseudo_mos | dnsmos_p808 | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2005.08138) |
| 17 | x | Packet Loss Concealment-related MOS Score (PLCMOS)  | pseudo_mos | plcmos | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2305.15127)|
| 18 |   | Virtual Speech Quality Objective Listener (VISQOL)  | visqol | visqol | [google-visqol](https://github.com/google/visqol) | [paper](https://arxiv.org/abs/2004.09584) |
| 19 | x | Speaker Embedding Similarity  | speaker | spk_similarity | [espnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2401.17230) |
| 20 | x | PESQ in TorchAudio-Squim  | squim_no_ref | torch_squim_pesq | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 21 | x | STOI in TorchAudio-Squim  | squim_no_ref | torch_squim_stoi | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 22 | x | SI-SDR in TorchAudio-Squim  | squim_no_ref | torch_squim_si_sdr | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 23 | x | MOS in TorchAudio-Squim  | squim_ref | torch_squim_mos |[torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 24 | x | Singing voice MOS  | singmos | singmos |[singmos](https://github.com/South-Twilight/SingMOS/tree/main) | [paper](https://arxiv.org/abs/2406.10911) |
| 25 | x | Log-Weighted Mean Square Error | log_wmse | log_wmse |[log_wmse](https://github.com/nomonosound/log-wmse-audio-quality) |
| 26 |   | Dynamic Time Warping Cost Metric | warpq | warpq |[WARP-Q](https://github.com/wjassim/WARP-Q) | [paper](https://arxiv.org/abs/2102.10449) |
| 27 | x | Sheet SSQA MOS Models | sheet_ssqa | sheet_ssqa |[Sheet](https://github.com/unilight/sheet/tree/main) | [paper](https://arxiv.org/abs/2411.03715) |
| 28 | x | ESPnet Speech Recognition-based Error Rate | espnet_wer | espnet_wer |[ESPnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/pdf/1804.00015) |
| 29 | x | ESPnet-OWSM Speech Recognition-based Error Rate | owsm_wer | owsm_wer |[ESPnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2309.13876) |
| 30 | x | OpenAI-Whisper Speech Recognition-based Error Rate | whisper_wer | whisper_wer |[Whisper](https://github.com/openai/whisper) | [paper](https://arxiv.org/abs/2212.04356) |
| 31 |   | UTMOSv2: UTokyo-SaruLab MOS Prediction System | utmosv2 | utmosv2 |[UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | [paper](https://arxiv.org/abs/2409.09305) |
| 32 |   | Speech Contrastive Regression for Quality Assessment with reference (ScoreQ) |  scoreq_ref | scoreq_ref |[ScoreQ](https://github.com/ftshijt/scoreq/tree/main) | [paper](https://arxiv.org/pdf/2410.06675) |
| 33 |   | Speech Contrastive Regression for Quality Assessment without reference (ScoreQ) | scoreq_nr | scoreq_nr |[ScoreQ](https://github.com/ftshijt/scoreq/tree/main) | [paper](https://arxiv.org/pdf/2410.06675) |
| 34 |   | Emotion2vec similarity (emo2vec) | emo2vec_similarity | emotion_similarity | [emo2vec](https://github.com/ftshijt/emotion2vec/tree/main) | [paper](https://arxiv.org/abs/2312.15185) | 
| 35 | x | Speech enhancement-based SI-SNR | se_snr | se_si_snr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 36 | x | Speech enhancement-based CI-SDR | se_snr | se_ci_sdr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 37 | x | Speech enhancement-based SAR | se_snr | se_sar | [ESPnet](https://github.com/espnet/espnet.git) | |
| 38 | x | Speech enhancement-based SDR | se_snr | se_sdr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 39 |   | NOMAD: Unsupervised Learning of Perceptual Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality Assessment |  nomad | nomad |[Nomad](https://github.com/shimhz/nomad/tree/main) | [paper](https://arxiv.org/abs/2309.16284) |
| 40 |   | Frechet Audio Distance (FAD) | fad | fad | [fadtk](https://github.com/microsoft/fadtk) | [paper](https://arxiv.org/abs/1812.08466) |
| 41 |   | Contrastive Language-Audio Pretraining Score (CLAP Score) | clap_score | clap_score | [fadtk](https://github.com/gudgud96/frechet-audio-distance) | [paper](https://arxiv.org/abs/2301.12661) |
| 42 |   | Audio Density and Coverage Score | audio_density_coverage | audio_density_coverage | [Sony-audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | [paper](https://arxiv.org/abs/2002.09797) |
| 43 |   | Accompaniment Prompt Adherence (APA) | apa | apa | [Sony-audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | [paper](https://arxiv.org/abs/2404.00775) |
| 44 |   | Kullback-Leibler Divergence on Embedding Distribution | kl_embedding | kl_embedding | [Stability-AI](https://github.com/Stability-AI/stable-audio-metrics) |  |
| 45 | x | PAM: Prompting Audio-Language Models for Audio Quality Assessment | pam | pam | [PAM](https://github.com/soham97/PAM/tree/main) | [Paper](https://arxiv.org/pdf/2402.00282)|
| 46 |  | Frequency-Weighted SEGmental SNR (FWSEGSNR) | pysepm | pysepm_fwsegsnr | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|
| 47 |  | Log Likelihood Ratio (LLR) | pysepm | pysepm_llr | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|
| 48 |  | Weighted Spectral Slope (WSS) | pysepm | pysepm_wss | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|
| 49 |  | Cepstrum Distance Objective Speech Quality Measure (CD) | pysepm | pysepm_cd | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ieeexplore.ieee.org/document/407206)|
| 50 |  | Composite Objective Speech Quality (composite) | pysepm | pysepm_Csig, pysepm_Cbak, pysepm_Covl | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|
| 51 |  | Coherence and speech intelligibility index (CSII) | pysepm | pysepm_csii_high, pysepm_csii_mid, pysepm_csii_low | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://www.researchgate.net/profile/James-Kates-2/publication/7842209_Coherence_and_the_speech_intelligibility_index/links/546f5dab0cf2d67fc0310f88/Coherence-and-the-speech-intelligibility-index.pdf)|
| 52 |  | Normalized-covariance measure (NCM) | pysepm | pysepm_ncm | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3037773/pdf/JASMAN-000128-003715_1.pdf)|
| 51 |  | Coherence and Speech Intelligibility Index (CSII) | pysepm | pysepm_csii_high, pysepm_csii_mid, pysepm_csii_low | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://www.researchgate.net/profile/James-Kates-2/publication/7842209_Coherence_and_the_speech_intelligibility_index/links/546f5dab0cf2d67fc0310f88/Coherence-and-the-speech-intelligibility-index.pdf)|
| 52 |  | Normalized-Covariance Measure (NCM) | pysepm | pysepm_ncm | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3037773/pdf/JASMAN-000128-003715_1.pdf)|
| 53 |  | Speech-to-Reverberation Modulation energy Ratio (SRMR) | srmr | srmr | [SRMRpy](https://github.com/shimhz/SRMRpy.git) | [Paper](http://www.individual.utoronto.ca/falkt/falk/pdf/FalkChan_TASLP2010.pdf)|
| 54 |  | Voice Activity Detection (VAD) | vad | vad_info | [SileroVAD](https://github.com/snakers4/silero-vad) | |
| 55 | x | AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks | asvspoof_score | asvspoof_score | [AASIST](https://github.com/clovaai/aasist/tree/main) | [Paper](https://ieeexplore.ieee.org/document/9747766)|
| 56 |  | NORESQA : A Framework for Speech Quality Assessment using Non-Matching References | noresqa | noresqa | [Noresqa](https://github.com/shimhz/Noresqa.git) | [Paper](https://proceedings.neurips.cc/paper/2021/file/bc6d753857fe3dd4275dff707dedf329-Paper.pdf)|
| A few more in verifying/progresss|


## Acknowledgement
We sincerely thank all the open-source implementations listed in https://github.com/shinjiwlab/versa/tree/main#list-of-metrics 
