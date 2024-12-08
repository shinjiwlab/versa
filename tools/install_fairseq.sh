#/bin/bash

if [ -d "fairseq" ]; then
    rm -rf fairseq
fi

git clone -b versa https://github.com/ftshijt/fairseq.git
cd fairseq

pip install -e .
cd ..
