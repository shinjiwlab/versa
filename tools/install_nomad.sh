#/bin/bash


rm -rf nomad

. ./install_nomad.sh > err.log 2>&1

# # NOTE(jiatong): a versa-specialized implementation for scoreq
git clone https://github.com/shimhz/nomad.git
cd nomad
pip install -e .
cd ..
