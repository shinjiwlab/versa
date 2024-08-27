#/bin/bash


rm -rf fairseq

git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install -e .
cd ..
