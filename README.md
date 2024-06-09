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

You will need to install visqol by yourself. Check https://github.com/google/visqol for details.
A installation bug for visqol for me is usually about lstdc++fs
Fix it by add 
```
build --linkopt=-lstdc++fs
```
to line 55 at .bazelrc in https://github.com/google/visqol


## Quick test
```
python speech_evaluation/bin/test.py
```

## Usage
```
python speech_evaluation/bin/scorer.py --score_config egs/codec_16k.yaml --gt egs/test/test1 --output_file egs/test/test_result.txt
```

Access `egs/*.yaml` for different config for differnt setups.
