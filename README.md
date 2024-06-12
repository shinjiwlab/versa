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
```
python speech_evaluation/bin/scorer.py --score_config egs/codec_16k.yaml --gt egs/test/test1 --output_file egs/test/test_result.txt
```

Access `egs/*.yaml` for different config for differnt setups.
