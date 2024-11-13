#/bin/bash


rm -rf UTMOSv2

git clone https://github.com/ftshijt/UTMOSv2.git
cd UTMOSv2
# Prevents LFS files from being temporarily downloaded during the installation process
GIT_LFS_SKIP_SMUDGE=1 pip install -e .
cd ..
