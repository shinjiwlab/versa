#/bin/bash


rm -rf fadtk

# NOTE(jiatong): a versa-specialized implementation for fadtk
git clone https://github.com/ftshijt/fadtk.git
cd fadtk
pip install -e .
cd ..
