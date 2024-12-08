#/bin/bash

if [ -d "nomad" ]; then
    rm -rf nomad
fi

. ./install_nomad.sh

# # NOTE(jiatong): a versa-specialized implementation for scoreq
git clone https://github.com/shimhz/nomad.git
cd nomad
pip install -e .
cd ..
