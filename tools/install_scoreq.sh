#/bin/bash


rm -rf WARP-Q

# # NOTE(jiatong): a versa-specialized implementation for scoreq
git clone https://github.com/ftshijt/scoreq.git
cd scoreq
pip install -e .
cd ..
