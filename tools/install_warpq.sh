#/bin/bash

if [ -d "WARP-Q" ]; then
    rm -rf WARP-Q
fi

# # NOTE(jiatong): a versa-specialized implementation for WARP-Q
git clone https://github.com/ftshijt/WARP-Q.git
cd WARP-Q
pip install -e .
cd ..
