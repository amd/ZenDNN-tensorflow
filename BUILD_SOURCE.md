
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

The repo defaults to the master development branch which doesn't has ZenDNN support. You need to check out a release branch to build, e.g. `r2.10_zendnn_rel` or `r2.12_zendnn_rel` etc.
```
$ git checkout branch_name  # r2.10_zendnn_rel, r2.12_zendnn_rel, etc.
```


### Set environment variables
Set environment variables for optimum performance. Some of the environment variables are for housekeeping purposes and can be ignored.
```
$ source scripts/zendnn_TF_env_setup.sh
```

### Only for TF2.12 Branch, Run the following script
bash setup_library.sh


### Export LD_LIBRARY_PATH
```
$ export LD_LIBRARY_PATH=bazel-out/k8-opt/bin/_solib_k8/_U_S_Sthird_Uparty_Szen_Udnn_Czen_Ulibs_Ulinux___Uexternal_Sllvm_Uopenmp/:bazel-out/k8-opt/bin/_solib_k8/_U_S_Sthird_Uparty_Szen_Udnn_Czen_Ulibs_Ulinux___Uexternal_Samd_Ublis/:bazel-out/k8-opt/bin/_solib_k8/_U@zen_Udnn_S_S_Czen_Udnn___Uexternal_Samd_Ulibm:$LD_LIBRARY_PATH
```

### Configure the build
Please run the ./configure script from the repository's root directory. This script will prompt you for the location of TensorFlow dependencies and asks for additional build configuration options (compiler flags, for example).
```
$ ./configure
You have bazel 6.1.0 installed.
Please specify the location of python. [Default is /Library/Frameworks/Python.framework/Versions/3.9/bin/python3]:


Found possible Python library paths:
  /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages
Please input the desired Python library path to use.  Default is [/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]:
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]:
No CUDA support will be enabled for TensorFlow.

Do you want to use Clang to build TensorFlow? [Y/n]:
Clang will be used to compile TensorFlow.

Please specify the path to clang executable. [Default is /usr/lib/llvm-16/bin/clang]:

You have Clang 16.0.4 installed.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]:


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Do you wish to build TensorFlow with iOS support? [y/N]: n
No iOS support will be enabled for TensorFlow.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
    --config=mkl            # Build with MKL support.
    --config=mkl_aarch64    # Build with oneDNN and Compute Library for the Arm Architecture (ACL).
    --config=monolithic     # Config for mostly static monolithic build.
    --config=numa           # Build with NUMA support.
    --config=dynamic_kernels    # (Experimental) Build kernels into separate shared objects.
    --config=v1             # Build with TensorFlow 1 API instead of TF 2 API.
Preconfigured Bazel build configs to DISABLE default on features:
    --config=nogcp          # Disable GCP support.
    --config=nonccl         # Disable NVIDIA NCCL support.
```

### Build and install the pip package
```
$ bazel clean --expunge
$ bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --config=zendnn --copt="-O3" -c opt --copt=-march='znver2'  //tensorflow/tools/pip_package:build_pip_package &> build_logs.txt
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ python -m pip install --force-reinstall --no-cache-dir /tmp/tensorflow_pkg/<your .whl file>
```

### Remove local host
```
cd $ZENDNN_PARENT_FOLDER
rm -rf host_zendnn/
cd $TF_GIT_ROOT
```
