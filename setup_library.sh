#This script does follwoing two task.
# 1. This script build AMD_BLIS and AMD_LIBM package.
# 2. Using local_host, these packages get downloaded in TensorFlow Repo.

cd $ZENDNN_PARENT_FOLDER
mkdir host_zendnn
git clone https://github.com/amd/aocl-libm-ose.git --quiet
cd aocl-libm-ose && git checkout -b temp origin/aocl-3.1 --quiet && scons -j32 --quiet && cd $ZENDNN_PARENT_FOLDER
zip -r host_zendnn/aocl-libm3.1.zip aocl-libm-ose/ --quiet
wget https://github.com/amd/blis/archive/refs/tags/4.1.tar.gz --quiet && tar -xf 4.1.tar.gz
cd blis-4.1 && make clean && make distclean && CFLAGS=-Wno-error && CC=gcc ./configure -a aocl_gemm --prefix=temp --enable-threading=openmp --enable-cblas amdzen &> $ZENDNN_PARENT_FOLDER/blis_log && make -j install &>> $ZENDNN_PARENT_FOLDER/blis_log && cd $ZENDNN_PARENT_FOLDER
tar -czf host_zendnn/blisv4.1.tar.gz blis-4.1/
rm -rf aocl-libm-ose/ blis-4.1/ 4.1.tar.gz

cd host_zendnn
zip_sum_libm=$(sha256sum aocl-libm3.1.zip | awk '{print $1}')
tar_sum_blis=$(sha256sum blisv4.1.tar.gz | awk '{print $1}')
nohup python3 -m http.server 8071 --bind 127.0.0.11 &>/dev/null &
host_pid=$!
#Copy LIBM tar sum and Path into tensorflow
sed -i "s/906391b8e35d95ff37f7bda7bbef3864609b58213e52aacf687a91bebcd617c0/$zip_sum_libm/" $TF_GIT_ROOT/tensorflow/workspace2.bzl
sed -i "s/aocl-libm-ose-aocl-3.1/aocl-libm-ose/" $TF_GIT_ROOT/tensorflow/workspace2.bzl
sed -i "s#https://github.com/amd/aocl-libm-ose/archive/refs/heads/aocl-3.1.zip#http://127.0.0.11:8071/aocl-libm3.1.zip#" $TF_GIT_ROOT/tensorflow/workspace2.bzl
#Copy BLIS tar sum and Path into tensorflow
sed -i "s/a05c6c7d359232580d1d599696053ad0beeedf50f3b88d5d22ee7d34375ab577/$tar_sum_blis/" $TF_GIT_ROOT/tensorflow/workspace2.bzl
sed -i "s#https://github.com/amd/blis/archive/refs/tags/4.1.tar.gz#http://127.0.0.11:8071/blisv4.1.tar.gz#" $TF_GIT_ROOT/tensorflow/workspace2.bzl
