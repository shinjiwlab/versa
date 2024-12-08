#/bin/bash

if [ -d "fadtk" ]; then
    rm -rf fadtk
fi

# NOTE(jiatong): a versa-specialized implementation for fadtk
git clone https://github.com/ftshijt/fadtk.git
cd fadtk
pip install -e .
cd ..
