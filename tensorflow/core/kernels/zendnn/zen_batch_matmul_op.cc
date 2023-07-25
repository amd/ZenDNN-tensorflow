/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include <vector>

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/zendnn/zen_matmul_ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"

using namespace zendnn;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {

// Returns the pair of dimensions along which to perform Tensor contraction to
// emulate matrix multiplication.
// For matrix multiplication of 2D Tensors X and Y, X is contracted along
// second dimension and Y is contracted along the first dimension (if neither X
// nor Y is adjointed). The dimension to contract along is switched when any
// operand is adjointed.
Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x, bool adj_y) {
  return Eigen::IndexPair<Eigen::DenseIndex>(adj_x ? 0 : 1, adj_y ? 1 : 0);
}

// Parallel batch matmul kernel based on the multi-threaded tensor contraction
// in Eigen.
template <typename Scalar>
struct ParallelMatMulKernel {
  static void Conjugate(const OpKernelContext *context, Tensor *out) {
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();
    auto z = out->tensor<Scalar, 3>();
    z.device(d) = z.conjugate();
  }

  static void Run(const OpKernelContext *context, const Tensor &in_x,
                  const Tensor in_y, bool adj_x, bool adj_y,
                  const MatMulBCast &bcast, Tensor *out, int start, int limit) {
    auto Tx = in_x.tensor<Scalar, 3>();
    auto Ty = in_y.tensor<Scalar, 3>();
    auto Tz = out->tensor<Scalar, 3>();
    // We use the identities
    //   conj(a) * conj(b) = conj(a * b)
    //   conj(a) * b = conj(a * conj(b))
    // to halve the number of cases. The final conjugation of the result is
    // done at the end of LaunchBatchMatMul<CPUDevice, Scalar>::Launch().
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x, adj_y);
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();

    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto &x_batch_indices = bcast.x_batch_indices();
    const auto &y_batch_indices = bcast.y_batch_indices();
    for (int64 i = start; i < limit; ++i) {
      const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;

      auto x = Tx.template chip<0>(x_batch_index);
      auto z = Tz.template chip<0>(i);
      if (adj_x != adj_y) {
        auto y = Ty.template chip<0>(y_batch_index).conjugate();
        z.device(d) = x.contract(y, contract_pairs);
      } else {
        auto y = Ty.template chip<0>(y_batch_index);
        z.device(d) = x.contract(y, contract_pairs);
      }
    }
  }
};

// Sequential batch matmul kernel that calls the regular Eigen matmul.
// We prefer this over the tensor contraction because it performs
// better on vector-matrix and matrix-vector products.
template <typename Scalar>
struct SequentialMatMulKernel {
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  static ConstMatrixMap ConstTensorSliceToEigenMatrix(const Tensor &t,
                                                      int slice) {
    return ConstMatrixMap(
        t.flat<Scalar>().data() + slice * t.dim_size(1) * t.dim_size(2),
        t.dim_size(1), t.dim_size(2));
  }

  static MatrixMap TensorSliceToEigenMatrix(Tensor *t, int slice) {
    return MatrixMap(
        t->flat<Scalar>().data() + slice * t->dim_size(1) * t->dim_size(2),
        t->dim_size(1), t->dim_size(2));
  }

  static void Run(const Tensor &in_x, const Tensor &in_y, bool adj_x,
                  bool adj_y, const MatMulBCast &bcast, Tensor *out, int start,
                  int limit) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto &x_batch_indices = bcast.x_batch_indices();
    const auto &y_batch_indices = bcast.y_batch_indices();
    for (int64 i = start; i < limit; ++i) {
      const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;
      auto x = ConstTensorSliceToEigenMatrix(in_x, x_batch_index);
      auto y = ConstTensorSliceToEigenMatrix(in_y, y_batch_index);
      auto z = TensorSliceToEigenMatrix(out, i);
      if (!adj_x) {
        if (!adj_y) {
          z.noalias() = x * y;
        } else {
          z.noalias() = x * y.adjoint();
        }
      } else {
        if (!adj_y) {
          z.noalias() = x.adjoint() * y;
        } else {
          z.noalias() = x.adjoint() * y.adjoint();
        }
      }
    }
  }
};
}  // namespace

template <typename Device, typename Scalar>
struct LaunchBatchMatMul;

template <typename Scalar>
struct LaunchBatchMatMul<CPUDevice, Scalar> {
  static void Launch(OpKernelContext *context, const Tensor &in_x,
                     const Tensor &in_y, bool adj_x, bool adj_y,
                     const MatMulBCast &bcast, Tensor *out) {
    typedef ParallelMatMulKernel<Scalar> ParallelMatMulKernel;
    bool conjugate_result = false;

    // Number of matrix multiplies i.e. size of the batch.
    const int64 batch_size = bcast.output_batch_size();
    const int64 cost_per_unit =
        in_x.dim_size(1) * in_x.dim_size(2) * out->dim_size(2);
    const int64 small_dim = std::min(
        std::min(in_x.dim_size(1), in_x.dim_size(2)), out->dim_size(2));
    const int64 kMaxCostOuterParallelism = 128 * 128 * 256;  // heuristic.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    if (small_dim > 1 &&
        (batch_size == 1 || cost_per_unit > kMaxCostOuterParallelism)) {
      // Parallelize over inner dims.
      // For large matrix products it is counter-productive to parallelize
      // over the batch dimension.
      ParallelMatMulKernel::Run(context, in_x, in_y, adj_x, adj_y, bcast, out,
                                0, batch_size);
      conjugate_result = adj_x;
    } else {
      // Parallelize over outer dims. For small matrices and large batches, it
      // is counter-productive to parallelize the inner matrix multiplies.
      Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
            cost_per_unit,
            [&in_x, &in_y, adj_x, adj_y, &bcast, out](int start, int limit) {
              SequentialMatMulKernel<Scalar>::Run(in_x, in_y, adj_x, adj_y,
                                                  bcast, out, start, limit);
            });
    }
    if (conjugate_result) {
      // We used one of the identities
      //   conj(a) * conj(b) = conj(a * b)
      //   conj(a) * b = conj(a * conj(b))
      // above, we need to conjugate the final output. This is a
      // no-op for non-complex types.
      ParallelMatMulKernel::Conjugate(context, out);
    }
  }
};

//  The third parameter v2_bcast is set to true if we are using V2 otherwise
//  we set it to false.
//  is_mul_add is set to true if we are fusing Mul and Add to BatchMatMul
template <typename Device, typename Scalar, bool v2_bcast, bool is_mul_add>
class ZenBatchMatMul : public OpKernel {
 public:
  explicit ZenBatchMatMul(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
  }

  virtual ~ZenBatchMatMul() {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor &lhs = ctx->input(0);
    const Tensor &rhs = ctx->input(1);
    bool is_float = std::is_same<Scalar, float>::value;
    if (is_float) {
      if (!v2_bcast) {
        // Using V1, so check to make sure lhs and rhs dimensions are correct
        // and no broadcasting is needed.
        OP_REQUIRES(ctx, lhs.dims() == rhs.dims(),
                    errors::InvalidArgument("lhs and rhs has different ndims: ",
                                            lhs.shape().DebugString(), " vs. ",
                                            rhs.shape().DebugString()));
        const int ndims = lhs.dims();
        OP_REQUIRES(
            ctx, ndims >= 2,
            errors::InvalidArgument("lhs and rhs ndims must be >= 2: ", ndims));
        for (int i = 0; i < ndims - 2; ++i) {
          OP_REQUIRES(ctx, lhs.dim_size(i) == rhs.dim_size(i),
                      errors::InvalidArgument(
                          "lhs.dim(", i, ") and rhs.dim(", i,
                          ") must be the same: ", lhs.shape().DebugString(),
                          " vs ", rhs.shape().DebugString()));
        }
      } else {
        OP_REQUIRES(
            ctx, lhs.dims() >= 2,
            errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims()));
        OP_REQUIRES(
            ctx, rhs.dims() >= 2,
            errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims()));
      }

      // lhs and rhs can have different dimensions
      const int ndims_lhs = lhs.dims();
      const int ndims_rhs = rhs.dims();

      // Get broadcast info
      MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
      OP_REQUIRES(
          ctx, bcast.IsValid(),
          errors::InvalidArgument(
              "In[0] and In[1] must have compatible batch dimensions: ",
              lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

      TensorShape out_shape = bcast.output_batch_shape();
      auto batch_size = bcast.output_batch_size();

      auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
      auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
      auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
      auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

      // TF-vanilla path will override lhs_reshaped and rhs_reshaped
      auto rhs_reshaped = rhs.template flat_inner_dims<float, 3>();
      auto lhs_reshaped = lhs.template flat_inner_dims<float, 3>();

      const uint64 M = lhs_reshaped.dimension(adj_x_ ? 2 : 1);
      const uint64 K = lhs_reshaped.dimension(adj_x_ ? 1 : 2);
      const uint64 N = rhs_reshaped.dimension(adj_y_ ? 1 : 2);

      // Switch for TF-vanilla and TF-zendnn
      // When M and N <= 64 TF-Vanilla path(Eigen implementation)
      //  is optimal for batched GEMM execution
      // ZenDNN kernel works well beyond above M and N values
      int switch_vanilla = (M <= 64) && (N <= 64);

      if (switch_vanilla) {
        Tensor lhs_reshaped;
        OP_REQUIRES(
            ctx,
            lhs_reshaped.CopyFrom(
                lhs, TensorShape({bcast.x_batch_size(), lhs_rows, lhs_cols})),
            errors::Internal("Failed to reshape In[0] from ",
                             lhs.shape().DebugString()));
        Tensor rhs_reshaped;
        OP_REQUIRES(
            ctx,
            rhs_reshaped.CopyFrom(
                rhs, TensorShape({bcast.y_batch_size(), rhs_rows, rhs_cols})),
            errors::Internal("Failed to reshape In[1] from ",
                             rhs.shape().DebugString()));
        // We need to reshape lhs and rhs before swapping their
        // rows and cols while reshapping (lhs and rhs) is not
        // needed in case of zendnn computation path.
        if (adj_x_) {
          std::swap(lhs_rows, lhs_cols);
        }
        if (adj_y_) {
          std::swap(rhs_rows, rhs_cols);
        }
        OP_REQUIRES(ctx, lhs_cols == rhs_rows,
                    errors::InvalidArgument(
                        "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows,
                        ": ", lhs.shape().DebugString(), " ",
                        rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_));

        out_shape.AddDim(lhs_rows);
        out_shape.AddDim(rhs_cols);

        Tensor *out = nullptr;
        // Implement Mempool later for Vanilla path too.
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
        if (out->NumElements() == 0) {
          return;
        }
        if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
          functor::SetZeroFunctor<Device, Scalar> f;
          f(ctx->eigen_device<Device>(), out->flat<Scalar>());
          return;
        }
        Tensor out_reshaped;
        OP_REQUIRES(ctx,
                    out_reshaped.CopyFrom(
                        *out, TensorShape({batch_size, lhs_rows, rhs_cols})),
                    errors::Internal("Failed to reshape output from ",
                                     out->shape().DebugString()));
        LaunchBatchMatMul<Device, Scalar>::Launch(ctx, lhs_reshaped,
                                                  rhs_reshaped, adj_x_, adj_y_,
                                                  bcast, &out_reshaped);

        if (is_mul_add) {
          const Tensor &mul_tensor = ctx->input(2);
          const Tensor &add_tensor = ctx->input(3);
          // Num of Attention Heads * SeqLength * SeqLength
          int out_d1d2d3 =
              out->dim_size(1) * out->dim_size(2) * out->dim_size(3);
          // SeqLength * SeqLength
          int out_d2d3 = out->dim_size(2) * out->dim_size(3);
          // Constant * SeqLength * SeqLength
          int add_d1d2d3 = add_tensor.dim_size(1) * add_tensor.dim_size(2) *
                           add_tensor.dim_size(3);
          float mul_node = mul_tensor.flat<float>().data()[0];
          const float *add_data = add_tensor.flat<float>().data();
          float *out_data = out->flat<float>().data();
          zendnnEnv zenEnvObj = readEnv();
          int no_of_threads = zenEnvObj.omp_num_threads;

// TODO: Implement Multi-level parallelism
// For Batch size < number of threads the current implementation
// will not perform optimally
#pragma omp parallel for num_threads(no_of_threads)
          for (int i = 0; i < out->dim_size(0); i++) {
            int bs_out_d1d2d3 = (i * out_d1d2d3);
            int bs_add_d1d2d3 = (i * add_d1d2d3);
            for (int j = 0; j < out->dim_size(1); j++) {
              int heads_out_d2d3 = bs_out_d1d2d3 + (j * out_d2d3);
              for (int k = 0; k < out->dim_size(2); k++) {
                int seq_out_d3 = heads_out_d2d3 + (k * out->dim_size(3));
                int seq_add_d3 = bs_add_d1d2d3 + (k * add_tensor.dim_size(3));
#pragma omp simd
                for (int l = 0; l < out->dim_size(3); l++) {
                  int index = seq_out_d3 + l;
                  int add_index = seq_add_d3 + l;
                  out_data[index] =
                      (out_data[index] * mul_node) + add_data[add_index];
                }
              }
            }
          }
        }
      } else {
        if (adj_x_) {
          std::swap(lhs_rows, lhs_cols);
        }
        if (adj_y_) {
          std::swap(rhs_rows, rhs_cols);
        }
        OP_REQUIRES(ctx, lhs_cols == rhs_rows,
                    errors::InvalidArgument(
                        "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows,
                        ": ", lhs.shape().DebugString(), " ",
                        rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_));

        out_shape.AddDim(lhs_rows);
        out_shape.AddDim(rhs_cols);
        // Update the output type
        zenTensorType out_type =
            (is_float) ? zenTensorType::FLOAT : zenTensorType::BFLOAT16;
        zendnnEnv zenEnvObj = readEnv();
        Tensor *out = nullptr;
        int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                               (ctx->expected_output_dtype(0) == DT_FLOAT ||
                                ctx->expected_output_dtype(0) == DT_BFLOAT16);
        ZenMemoryPool<Scalar> *zenPoolBuffer = NULL;

        // ZenMemPool Optimization reuse o/p tensors from the pool. By default
        //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
        //  pool optimization
        //  Cases where tensors in pool are not free or requested size is more
        //  than available tensor size in Pool, control will fall back to
        //  default way of allocation i.e. with allocate_output(..)
        if (zenEnableMemPool) {
          unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
          zenPoolBuffer = ZenMemoryPool<Scalar>::getZenMemPool(threadID);
          if (zenPoolBuffer) {
            int status = zenPoolBuffer->acquireZenPoolTensor(
                ctx, &out, out_shape, out_links, reset, out_type);
            if (status) {
              zenEnableMemPool = false;
            }
          } else {
            zenEnableMemPool = false;
          }
        }
        if (!zenEnableMemPool) {
          OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
        }

        if (out->NumElements() == 0) {
          return;
        }
        if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
          functor::SetZeroFunctor<Device, Scalar> f;
          f(ctx->eigen_device<Device>(), out->flat<Scalar>());
          return;
        }

        auto out_reshaped = out->template flat_inner_dims<float, 3>();

        std::vector<int> m_array(batch_size, M);
        std::vector<int> n_array(batch_size, N);
        std::vector<int> k_array(batch_size, K);
        std::vector<int> lda_array(batch_size, adj_x_ ? M : K);
        std::vector<int> ldb_array(batch_size, adj_y_ ? K : N);
        std::vector<int> ldc_array(batch_size, N);
        std::vector<float> alpha_array(batch_size, 1.0);
        std::vector<float> beta_array(batch_size, 0.0);
        std::vector<int> group_size(1, batch_size);
        std::vector<const float *> a_array;
        std::vector<const float *> b_array;
        std::vector<float *> c_array;
        std::vector<const float *> add_array;

        a_array.reserve(batch_size);
        b_array.reserve(batch_size);
        c_array.reserve(batch_size);
        add_array.reserve(out->dim_size(0));

        if (!bcast.IsBroadcastingRequired()) {
          for (int64 i = 0; i < batch_size; i++) {
            a_array.push_back(&lhs_reshaped(i, 0, 0));
            b_array.push_back(&rhs_reshaped(i, 0, 0));
            c_array.push_back(&out_reshaped(i, 0, 0));
          }
        } else {
          // Broadcasting is needed, so get the mapping from flattened output
          // batch indices to x's and y's flattened batch indices.
          const std::vector<int64> &a_batch_indices = bcast.x_batch_indices();
          const std::vector<int64> &b_batch_indices = bcast.y_batch_indices();

          for (int64 i = 0; i < batch_size; i++) {
            a_array.push_back(&lhs_reshaped(a_batch_indices[i], 0, 0));
            b_array.push_back(&rhs_reshaped(b_batch_indices[i], 0, 0));
            c_array.push_back(&out_reshaped(i, 0, 0));
          }
        }
        float mul_node = 1;
        if (is_mul_add) {
          const Tensor &mul_tensor = ctx->input(2);
          const Tensor &add_tensor = ctx->input(3);
          mul_node = mul_tensor.flat<float>().data()[0];
          auto add_reshaped = add_tensor.template flat_inner_dims<float, 3>();
          for (int64 i = 0; i < out->dim_size(0); i++) {
            add_array.push_back(&add_reshaped(i, 0, 0));
          }
        }
        bool cblasRowMajor = 1;
        zenBatchMatMul(cblasRowMajor, adj_x_, adj_y_, &m_array[0], &n_array[0],
                       &k_array[0], &alpha_array[0], &a_array[0], &lda_array[0],
                       &b_array[0], &ldb_array[0], &beta_array[0], &c_array[0],
                       &ldc_array[0], 1, &group_size[0], is_mul_add,
                       &add_array[0], mul_node, out->dim_size(0));

        // If ZenMemPool Optimization is enabled(default), update the state of
        //  Memory pool based on input_array address
        if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
          zenPoolBuffer->zenMemPoolFree(ctx, (void *)a_array[0]);
          zenPoolBuffer->zenMemPoolFree(ctx, (void *)b_array[0]);
        }
      }
    }
    // Batch matmul for BF16
    else {
      ZenExecutor *ex = ex->getInstance();
      engine eng = ex->getEngine();
      stream s = ex->getStream();
      using tag = memory::format_tag;
      using dt = memory::data_type;
      std::vector<primitive> net;
      std::vector<std::unordered_map<int, memory>> net_args;
      // BF16 kernel only accepts 4D tensors
      // since it's hardcoded as 4D and only tested with 4D tensors
      auto input_map = lhs.tensor<Scalar, 4>();
      const Scalar *input_array = input_map.data();
      auto filter_map = rhs.tensor<Scalar, 4>();
      const Scalar *filter_array = filter_map.data();
      Scalar *in_arr = const_cast<Scalar *>(input_array);
      Scalar *filt_arr = const_cast<Scalar *>(filter_array);
      MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
      OP_REQUIRES(
          ctx, bcast.IsValid(),
          errors::InvalidArgument(
              "In[0] and In[1] must have compatible batch dimensions: ",
              lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));
      auto rhs_reshaped = rhs.template flat_inner_dims<Scalar, 3>();
      auto lhs_reshaped = lhs.template flat_inner_dims<Scalar, 3>();
      const uint64 M = lhs_reshaped.dimension(adj_x_ ? 2 : 1);
      const uint64 K = lhs_reshaped.dimension(adj_x_ ? 1 : 2);
      const uint64 N = rhs_reshaped.dimension(adj_y_ ? 1 : 2);

      TensorShape out_shape = bcast.output_batch_shape();
      out_shape.AddDim(M);
      out_shape.AddDim(N);
      auto batch_size = bcast.output_batch_size();
      zenTensorType out_type =
          (is_float) ? zenTensorType::FLOAT : zenTensorType::BFLOAT16;
      zendnnEnv zenEnvObj = readEnv();
      Tensor *output = nullptr;
      int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                             (ctx->expected_output_dtype(0) == DT_FLOAT ||
                              ctx->expected_output_dtype(0) == DT_BFLOAT16);
      ZenMemoryPool<Scalar> *zenPoolBuffer = NULL;
      if (zenEnableMemPool) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<Scalar>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          int status = zenPoolBuffer->acquireZenPoolTensor(
              ctx, &output, out_shape, out_links, reset, out_type);
          if (status) {
            zenEnableMemPool = false;
          }
        } else {
          zenEnableMemPool = false;
        }
      }
      if (!zenEnableMemPool) {
        // Output tensor is of the following dimensions:
        // [ in_batch, out_rows, out_cols, out_depth ]
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
      }
      auto output_map = output->tensor<Scalar, 4>();
      Scalar *output_array = const_cast<Scalar *>(output_map.data());

      memory::dims src_dims = {lhs.dim_size(0), lhs.dim_size(1), M, K};
      memory::dims weight_dims = {rhs.dim_size(0), rhs.dim_size(1), K, N};
      memory::dims dst_dims = {lhs.dim_size(0), lhs.dim_size(1), M, N};

      memory::desc src_md = memory::desc({src_dims}, dt::bf16, tag::abcd);
      memory::desc dst_md = memory::desc({dst_dims}, dt::bf16, tag::abcd);
      memory::desc matmul_weights_md =
          memory::desc({weight_dims}, dt::bf16, adj_y_ ? tag::abdc : tag::abcd);
      zendnn::memory user_weights_memory, src_memory, dst_memory;
      src_memory = memory({{src_dims}, dt::bf16, tag::abcd}, eng, in_arr);
      dst_memory = memory({{dst_dims}, dt::bf16, tag::abcd}, eng, output_array);

      user_weights_memory =
          memory({{weight_dims}, dt::bf16, adj_y_ ? tag::abdc : tag::abcd}, eng,
                 filt_arr);
      primitive_attr matmul_attr;

      // Binary postop support
      if (is_mul_add) {
        const Tensor &mul_tensor = ctx->input(2);
        const Tensor &add_tensor = ctx->input(3);
        const Scalar *mul_node = mul_tensor.flat<Scalar>().data();
        Scalar *mul_arr = const_cast<Scalar *>(mul_node);
        const Scalar *add_data = add_tensor.flat<Scalar>().data();
        Scalar *add_arr = const_cast<Scalar *>(add_data);
        memory::dims mul_dims = {1, 1, 1, 1};
        memory::dims add_dims = {add_tensor.dim_size(0), add_tensor.dim_size(1),
                                 add_tensor.dim_size(2),
                                 add_tensor.dim_size(3)};
        zendnn::post_ops post_ops;
        post_ops.append_binary(algorithm::binary_mul,
                               memory::desc({mul_dims}, dt::bf16, tag::abcd));
        post_ops.append_binary(algorithm::binary_add,
                               memory::desc({add_dims}, dt::bf16, tag::abcd));
        matmul_attr.set_post_ops(post_ops);
        matmul::desc matmul_pd1 =
            matmul::desc(src_md, matmul_weights_md, dst_md);
        matmul::primitive_desc matmul_pd =
            matmul::primitive_desc(matmul_pd1, matmul_attr, eng);
        net.push_back(matmul(matmul_pd));
        zendnn::memory postop_memory1, postop_memory2;
        postop_memory1 =
            memory({{mul_dims}, dt::bf16, tag::abcd}, eng, mul_arr);
        postop_memory2 =
            memory({{add_dims}, dt::bf16, tag::abcd}, eng, add_arr);
        net_args.push_back(
            {{ZENDNN_ARG_SRC, src_memory},
             {ZENDNN_ARG_WEIGHTS, user_weights_memory},
             {ZENDNN_ARG_DST, dst_memory},
             {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1,
              postop_memory1},
             {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1,
              postop_memory2}});
      } else {
        matmul::desc matmul_pd1 =
            matmul::desc(src_md, matmul_weights_md, dst_md);
        matmul::primitive_desc matmul_pd =
            matmul::primitive_desc(matmul_pd1, matmul_attr, eng);

        net.push_back(matmul(matmul_pd));
        net_args.push_back({{ZENDNN_ARG_SRC, src_memory},
                            {ZENDNN_ARG_WEIGHTS, user_weights_memory},
                            {ZENDNN_ARG_DST, dst_memory}});
      }
      assert(net.size() == net_args.size() && "something is missing");
      for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(s, net_args.at(i));
      }
      if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
        zenPoolBuffer->zenMemPoolFree(ctx, (void *)input_array);
        zenPoolBuffer->zenMemPoolFree(ctx, (void *)filter_array);
      }
    }
  }

 private:
  bool adj_x_;
  bool adj_y_;
  bool reset;
  int in_links, out_links;
};

#define REGISTER_BATCH_MATMUL_ZEN(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ZenBatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      ZenBatchMatMul<CPUDevice, TYPE, false, false>);                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ZenBatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenBatchMatMul<CPUDevice, TYPE, true, false>);                          \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedBatchMatMulV2")                      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T"),                     \
                          ZenBatchMatMul<CPUDevice, TYPE, true, true>);

TF_CALL_float(REGISTER_BATCH_MATMUL_ZEN)
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_ZEN)
#undef REGISTER_BATCH_MATMUL_ZEN

}  // end namespace tensorflow
