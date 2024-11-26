#/bin/bash


rm -rf fairseq

git clone https://github.com/espnet/fairseq.git
cd fairseq
pip install -e .
cd ..
