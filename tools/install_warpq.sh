#/bin/bash


rm -rf WARP-Q

# # NOTE(jiatong): a versa-specialized implementation for WARP-Q
git clone https://github.com/ftshijt/WARP-Q.git
cd WARP-Q
pip install -e .
cd ..
