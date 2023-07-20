#*******************************************************************************
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#*******************************************************************************

#!/bin/bash

#-----------------------------------------------------------------------------
#   zendnn_TF_env_setup.sh
#   Prerequisite: This script needs to run first to setup environment variables
#                 before TensorFlow GCC build
#
#   This script does following:
#   -Checks if important env variables are declared
#   -Checks and print version informations for following:
#       -make, gcc, g++, ld, python
#   -Sets important environment variables for benchmarking:
#       -OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PROC_BIND
#   -Calls script to gather HW, OS, Kernel, Bios information
#----------------------------------------------------------------------------

#Function to check mandatory prerequisites
function check_mandatory_prereqs() {
    if type -P make >/dev/null 2>&1;
    then
        echo "make is installed"
        echo `make -v | grep "GNU Make"`
    else
        echo "make is not installed, install make"
        return
    fi

    if type -P gcc >/dev/null 2>&1;
    then
        echo "gcc is installed"
        echo `gcc --version | grep "gcc "`
    else
        echo "gcc is not installed, install gcc"
        return
    fi

    if type -P g++ >/dev/null 2>&1;
    then
        echo "g++ is installed"
        echo `g++ --version | grep "g++ "`
    else
        echo "g++ is not installed, install g++"
        return
    fi

    if type -P ld >/dev/null 2>&1;
    then
        echo "ld is installed"
        echo `ld --version | grep "GNU ld "`
    else
        echo "ld is not installed, install ld"
        return
    fi

    if type -P python3 >/dev/null 2>&1;
    then
        echo "python3 is installed"
        echo `python3 --version`
    else
        echo "python3 is not installed, install python3"
        return
    fi
}

#Function to check optional prerequisites
function check_optional_prereqs() {
    if type -P lscpu >/dev/null 2>&1;
    then
        echo "lscpu is installed"
        echo `lscpu --version`
    else
        echo "lscpu is not installed, install lscpu"
    fi

    # Check if hwloc/lstopo-no-graphics is installed
    if type -P lstopo-no-graphics >/dev/null 2>&1;
    then
        echo "lstopo-no-graphics is installed"
        echo `lstopo-no-graphics --version`
    else
        echo "lstopo-no-graphics is not installed, install hwloc/lstopo-no-graphics"
    fi

    # Check if uname is installed
    if type -P uname >/dev/null 2>&1;
    then
        echo "uname is installed"
        echo `uname --version`
    else
        echo "uname is not installed, install uname"
    fi

    # Check if dmidecode is installed
    if type -P dmidecode >/dev/null 2>&1;
    then
        echo "dmidecode is installed"
        echo `dmidecode --version`
    else
        echo "dmidecode is not installed, install dmidecode"
    fi
}

#------------------------------------------------------------------------------
# Check if mandatory prerequisites are installed
echo "Checking mandatory prerequisites"
check_mandatory_prereqs

echo "Checking optional prerequisites"
# Check if optional prerequisites are installed
check_optional_prereqs
echo""

#------------------------------------------------------------------------------
if [ -z "$ZENDNN_LOG_OPTS" ];
then
    export ZENDNN_LOG_OPTS=ALL:0
    echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
else
    echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
fi

if [ -z "$OMP_NUM_THREADS" ];
then
    export OMP_NUM_THREADS=96
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
else
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
fi

if [ -z "$OMP_WAIT_POLICY" ];
then
    export OMP_WAIT_POLICY=ACTIVE
    echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
else
    echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
fi

if [ -z "$OMP_PROC_BIND" ];
then
    export OMP_PROC_BIND=FALSE
    echo "OMP_PROC_BIND=$OMP_PROC_BIND"
else
    echo "OMP_PROC_BIND=$OMP_PROC_BIND"
fi

#By default setting ZENDNN_TF_VERSION as 2.12
export ZENDNN_TF_VERSION=2.12
echo "ZENDNN_TF_VERSION=$ZENDNN_TF_VERSION"

#Disabling export of GCC BLIS and ZenDNN library and other paths when building
#TensorFlow with --config=zendnn option
if [ -z "$ZENDNN_TF_GCC_CONFIG_BUILD" ];
then
    export ZENDNN_TF_GCC_CONFIG_BUILD=1
fi
echo "ZENDNN_TF_GCC_CONFIG_BUILD=$ZENDNN_TF_GCC_CONFIG_BUILD"

#Use local copy of ZenDNN library source code when building
#Tensorflow with --config=zendnn
if [ -z "$ZENDNN_TF_USE_LOCAL_ZENDNN" ];
then
    export ZENDNN_TF_USE_LOCAL_ZENDNN=1
fi
echo "ZENDNN_TF_USE_LOCAL_ZENDNN=$ZENDNN_TF_USE_LOCAL_ZENDNN"

#Use local copy of Blis source code when building
#Tensorflow with --config=zendnn
if [ -z "$ZENDNN_TF_USE_CUSTOM_BLIS" ];
then
    export ZENDNN_TF_USE_CUSTOM_BLIS=0
    export ZENDNN_TF_CUSTOM_BLIS="blis"
fi
echo "ZENDNN_TF_USE_CUSTOM_BLIS=$ZENDNN_TF_USE_CUSTOM_BLIS"
echo "ZENDNN_TF_CUSTOM_BLIS=$ZENDNN_TF_CUSTOM_BLIS"

#If the environment variable OMP_DYNAMIC is set to true, the OpenMP implementation
#may adjust the number of threads to use for executing parallel regions in order
#to optimize the use of system resources. ZenDNN depend on a number of threads
#which should not be modified by runtime, doing so can cause incorrect execution
export OMP_DYNAMIC=FALSE
echo "OMP_DYNAMIC=$OMP_DYNAMIC"

#Disable TF check for training ops and stop execution if any training ops
#found in TF graph. By default, its enabled
export ZENDNN_INFERENCE_ONLY=1
echo "ZENDNN_INFERENCE_ONLY=$ZENDNN_INFERENCE_ONLY"

#Disable TF memory pool optimization, By default, its enabled
export ZENDNN_ENABLE_MEMPOOL=1
echo "ZENDNN_ENABLE_MEMPOOL=$ZENDNN_ENABLE_MEMPOOL"

#Set the max no. of tensors that can be used inside TF memory pool, Default is
#set to 1024
export ZENDNN_TENSOR_POOL_LIMIT=1024
echo "ZENDNN_TENSOR_POOL_LIMIT=$ZENDNN_TENSOR_POOL_LIMIT"

#Enable fixed max size allocation for Persistent tensor with TF memory pool
#optimization, By default, its disabled
export ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=0
echo "ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=$ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE"

# Convolution GEMM Algo path
export ZENDNN_CONV_ALGO=4
echo "ZENDNN_CONV_ALGO=$ZENDNN_CONV_ALGO"

# INT8 support  is disabled by default
export ZENDNN_INT8_SUPPORT=0
echo "ZENDNN_INT8_SUPPORT=$ZENDNN_INT8_SUPPORT"

# ZENDNN_GEMM_ALGO is set to 3 by default
export ZENDNN_GEMM_ALGO=3
echo "ZENDNN_GEMM_ALGO=$ZENDNN_GEMM_ALGO"

# Switch to enable Conv, Add fusion on users discretion. Currently it is
# safe to enable this switch for resnet50v1_5, resnet101, and
# inception_resnet_v2 models only. By default the switch is disabled.
export ZENDNN_TF_CONV_ADD_FUSION_SAFE=0
echo "ZENDNN_TF_CONV_ADD_FUSION_SAFE=$ZENDNN_TF_CONV_ADD_FUSION_SAFE"

# Primitive reuse is disabled by default
export TF_ZEN_PRIMITIVE_REUSE_DISABLE=FALSE
echo "TF_ZEN_PRIMITIVE_REUSE_DISABLE=$TF_ZEN_PRIMITIVE_REUSE_DISABLE"

#By default ZenDNN optimization are enabled
export TF_ENABLE_ZENDNN_OPTS=1
echo "TF_ENABLE_ZENDNN_OPTS=$TF_ENABLE_ZENDNN_OPTS"

# Primitive Caching Capacity
export ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY: $ZENDNN_PRIMITIVE_CACHE_CAPACITY"

# MAX_CPU_ISA
# MAX_CPU_ISA is disabld at build time. When feature is enabled, uncomment the
# below 2 lines
#export ZENDNN_MAX_CPU_ISA=ALL
#echo "ZENDNN_MAX_CPU_ISA: $ZENDNN_MAX_CPU_ISA"

# Enable primitive create and primitive execute logs. By default it is disabled
export ZENDNN_PRIMITIVE_LOG_ENABLE=0
echo "ZENDNN_PRIMITIVE_LOG_ENABLE: $ZENDNN_PRIMITIVE_LOG_ENABLE"

# Default location for benchmark data.
export ZENDNN_TEST_USE_COMMON_BENCHMARK_LOC=FALSE
echo "ZENDNN_TEST_USE_COMMON_BENCHMARK_LOC: $ZENDNN_TEST_USE_COMMON_BENCHMARK_LOC"
export ZENDNN_TEST_COMMON_BENCHMARK_LOC=/home/amd/benchmark_data
echo "ZENDNN_TEST_COMMON_BENCHMARK_LOC: $ZENDNN_TEST_COMMON_BENCHMARK_LOC"

echo "LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

# By default build Open sourced TensorFlow and ZenDNN
export ZENDNN_TF_USE_OPENSOURCE=1

#-------------------------------------------------------------------------------
# Path related details
#-----------------------------------------------------------------------------
#Check if below declaration of TF_GIT_ROOT is correct
export TF_GIT_ROOT=$(pwd)
if [ -z "$TF_GIT_ROOT" ];
then
    echo "Error: Environment variable TF_GIT_ROOT needs to be set"
    echo "Error: \$TF_GIT_ROOT points to root of ZenDNN_TensorFlow repo"
else
    [ ! -d "$TF_GIT_ROOT" ] && echo "Directory ZenDNN_TensorFlow DOES NOT exists!"
    echo "TF_GIT_ROOT: $TF_GIT_ROOT"
fi

#Change ZENDNN_PARENT_FOLDER as per need in future
#Current assumption, TF is located parallel to ZenDNN
cd ..
export ZENDNN_PARENT_FOLDER=$(pwd)
cd -

if [ -z "$ZENDNN_PARENT_FOLDER" ];
then
    echo "Error: Environment variable ZENDNN_PARENT_FOLDER needs to be set"
else
    echo "ZENDNN_PARENT_FOLDER: $ZENDNN_PARENT_FOLDER"
fi

#Check if below declaration of ZENDNN_GIT_ROOT is correct
export ZENDNN_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN
if [ -z "$ZENDNN_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_GIT_ROOT needs to be set"
    echo "Error: \$ZENDNN_GIT_ROOT points to root of ZENDNN repo"
else
    [ ! -d "$ZENDNN_GIT_ROOT" ] && echo "Directory ZenDNN DOES NOT exists!"
    echo "ZENDNN_GIT_ROOT: $ZENDNN_GIT_ROOT"
fi

#Change ZENDNN_UTILS_GIT_ROOT as per need in future
cd ..
export ZENDNN_UTILS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN_utils
cd -
if [ -z "$ZENDNN_UTILS_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_UTILS_GIT_ROOT needs to be set"
else
    [ ! -d "$ZENDNN_UTILS_GIT_ROOT" ] && echo "Directory ZenDNN_utils DOES NOT exists!"
    echo "ZENDNN_UTILS_GIT_ROOT: $ZENDNN_UTILS_GIT_ROOT"
fi

#Change ZENDNN_TOOLS_GIT_ROOT as per need in future
cd ..
export ZENDNN_TOOLS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN_tools
cd -
if [ -z "$ZENDNN_UTILS_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_TOOLS_GIT_ROOT needs to be set"
else
    [ ! -d "$ZENDNN_TOOLS_GIT_ROOT" ] && echo "Directory ZenDNN_tools DOES NOT exists!"
    echo "ZENDNN_TOOLS_GIT_ROOT: $ZENDNN_TOOLS_GIT_ROOT"
fi

export BENCHMARKS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/benchmarks
echo "BENCHMARKS_GIT_ROOT: $BENCHMARKS_GIT_ROOT"


#-------------------------------------------------------------------------------
# HW, HW architecture, Cache, OS, Kernel details
#-----------------------------------------------------------------------------
# Go to ZENDNN_GIT_ROOT
cd $TF_GIT_ROOT

chmod u+x scripts/gather_hw_os_kernel_bios_info.sh
echo "scripts/gather_hw_os_kernel_bios_info.sh"
source scripts/gather_hw_os_kernel_bios_info.sh true > system_hw_os_kernel_bios_info.txt

#-------------------------------------------------------------------------------
# Go to TF_GIT_ROOT
cd $TF_GIT_ROOT
echo -e "\n"
echo "Please set below environment variables explicitly as per the platform you are using!!"
echo -e "\tOMP_NUM_THREADS"
echo "Please set below environment variables explicitly for better performance!!"
echo "OMP_PROC_BIND=CLOSE"
echo "OMP_PLACES=CORES"
echo -e "\n"
