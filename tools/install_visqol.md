## Step-by-step visqol installiation

### get visqol repository:
`git clone https://github.com/google/visqol`

### Install Bazel (take version 5.1.0 as an example):
```
wget https://github.com/bazelbuild/bazel/releases/download/5.1.0/bazel-5.1.0-installer-linux-x86_64.sh 
chmod +x bazel-5.1.0-installer-linux-x86_64.sh
./bazel-version-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"
export PATH="$PATH:$HOME/.bashrc"
export PATH="$PATH:$HOME/.zshrc"
```

### revise compile config and compile. You may need 32G memory for this stage
1. add `build --linkopt=-lstdc++fs` after line 55 of `.bazelrc`
2. replace the version to `5.1.0` in `.bazelversion`
3. update `WORKSPACE` to new armadillo version as suggested in https://github.com/google/visqol/pull/119/files
   - for additional note, 10.1.0 is also deprecated. You may consider using
  ```
    sha256 = "023242fd59071d98c75fb015fd3293c921132dc39bf46d221d4b059aae8d79f4",
    strip_prefix = "armadillo-14.4.0",
    urls = ["http://sourceforge.net/projects/arma/files/armadillo-14.4.0.tar.xz"],
  ``` 
4. compile with `bazel build :visqol -c opt`

### install in python 
`pip install .`
