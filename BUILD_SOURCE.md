
**`Documentation`** |
------------------- |
To build TensorFlow with ZenDNN follow below steps.

## Build From Source
### Setup for Linux
Create and activate a conda environment and install the following dependencies
```
$ pip install --upgrade --no-deps --force-reinstall --no-cache-dir numpy absl-py
$ pip install -U pip six wheel setuptools mock future scons requests packaging
$ pip install -U keras_applications --no-deps
$ pip install -U keras_preprocessing --no-deps
```


### Install Bazel
To build TensorFlow, you will need to install Bazel. [Bazelisk](https://github.com/bazelbuild/bazelisk) is an easy way to install Bazel and automatically downloads the correct Bazel version for TensorFlow. For ease of use, add Bazelisk as the **bazel** executable in your **PATH**.

If Bazelisk is not available, you can manually [install Bazel](https://bazel.build/install). Make sure to install the correct Bazel version from TensorFlow's [.bazelversion](https://github.com/tensorflow/tensorflow/blob/master/.bazelversion) file.


### Download the AMD ZenDNN TensorFlow source code
Location of AMD ZenDNN TensorFlow: [AMD ZenDNN TensorFlow](https://github.com/amd/ZenDNN-tensorflow).

Checkout AMD ZenDNN TensorFlow
```
$ git clone https://github.com/amd/ZenDNN-tensorflow.git
$ cd  ZenDNN-tensorflow
```

The repo defaults to the master development branch which doesn't have ZenDNN support. You need to check out a release branch to build, e.g. `r2.10_zendnn_rel` or `r2.12_zendnn_rel` etc.
```
$ git checkout branch_name  # r2.10_zendnn_rel, r2.12_zendnn_rel, etc.
```


### Set environment variables
Set environment variables for optimum performance. Some of the environment variables are for housekeeping purposes and can be ignored.
```
$ source scripts/zendnn_TF_env_setup.sh
```


### Export LD_LIBRARY_PATH
```
$ export LD_LIBRARY_PATH=bazel-out/k8-opt/bin/_solib_k8/_U_S_Sthird_Uparty_Szen_Udnn_Czen_Ulibs_Ulinux___Uexternal_Sllvm_Uopenmp/:bazel-out/k8-opt/bin/_solib_k8/_U_S_Sthird_Uparty_Szen_Udnn_Czen_Ulibs_Ulinux___Uexternal_Samd_Ublis/:bazel-out/k8-opt/bin/_solib_k8/_U@zen_Udnn_S_S_Czen_Udnn___Uexternal_Samd_Ulibm:$LD_LIBRARY_PATH
```

### Configure the build
Please run the ./configure script from the repository's root directory. This script will prompt you for the location of TensorFlow dependencies and asks for additional build configuration options (compiler flags, for example).
```
$ ./configure
```

### Build and install the pip package
```
$ bazel clean --expunge
$ bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --config=zendnn --copt="-O3" -c opt --copt=-march='znver2'  //tensorflow/tools/pip_package:build_pip_package &> build_logs.txt
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ python -m pip install --force-reinstall --no-cache-dir /tmp/tensorflow_pkg/<your .whl file>
```
