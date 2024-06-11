### get visqol repository:
`git clone https://github.com/google/visqol`

### Install Bazel (take version 5.1.0 as an example):
`wget https://github.com/bazelbuild/bazel/releases/download/5.1.0/bazel-5.1.0-installer-linux-x86_64.sh` 
`bash bazel-5.1.0-installer-linux-x86_64.sh --user`

### revise compile config and compile. You may need 32G memory for this stage
(1) add `build --linkopt=-lstdc++fs` after line 55 of `.bazelrc`
(2) replace the version to `5.1.0` in `.bazelversion`
(3) compile with `bazel build :visqol -c opt`

### install in python 
`pip install .`
