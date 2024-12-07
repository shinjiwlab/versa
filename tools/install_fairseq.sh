#/bin/bash


rm -rf fairseq

git clone -b versa https://github.com/ftshijt/fairseq.git
cd fairseq

pip install -e .
cd ..
