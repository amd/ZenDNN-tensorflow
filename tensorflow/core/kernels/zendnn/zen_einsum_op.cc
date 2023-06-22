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

#include "tensorflow/core/kernels/linalg/einsum_op_impl.h"
#include "tensorflow/core/kernels/zendnn/zen_matmul_ops_util.h"

using namespace zendnn;
using namespace std;

namespace tensorflow {

struct ZenEinsumHelper {
  // Contracts the inputs along the last axis. (or the second last if the
  // corresponding value of swap_free_and_contract is true). The batch
  // dimensions are broadcast to the output shape.

  template <typename Device, typename T>
  static Status ZenContractOperands(
      OpKernelContext *ctx, absl::Span<const Tensor> inputs,
      absl::Span<const bool> swap_free_and_contract, Tensor *output,
      int out_links, bool reset) {

    if (inputs.size() == 1)
      return EinsumHelper::CopyFrom(inputs[0], inputs[0].shape(), output);
    MatMulBCast bcast(inputs[0].shape().dim_sizes(),
                      inputs[1].shape().dim_sizes());

    Tensor lhs = inputs[0];
    Tensor rhs = inputs[1];

    TensorShape output_shape = bcast.output_batch_shape();
    for (int i = 0; i < inputs.size(); ++i) {
      const int64 free_axis =
          inputs[i].dims() - (swap_free_and_contract[i] ? 1 : 2);
      output_shape.AddDim(inputs[i].dim_size(free_axis));
    }

    bool trans_x = swap_free_and_contract[0];
    bool trans_y = !swap_free_and_contract[1];
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));

    if (!(lhs.dims() >= 2))
      return errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims());

    if (!(rhs.dims() >= 2))
      return errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims());

    const auto ndims_lhs = lhs.dims();
    const auto ndims_rhs = rhs.dims();
    // In[0] and In[1] must have compatible batch dimensions
    if (!(bcast.IsValid()))
      return errors::InvalidArgument(
          "In[0] and In[1] must have compatible batch dimensions: ",
          lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString());

    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    auto rhs_reshaped = rhs.template flat_inner_dims<T, 3>();
    auto lhs_reshaped = lhs.template flat_inner_dims<T, 3>();

    const uint64 M = lhs_reshaped.dimension(trans_x ? 2 : 1);
    const uint64 K = lhs_reshaped.dimension(trans_x ? 1 : 2);
    const uint64 N = rhs_reshaped.dimension(trans_y ? 1 : 2);

    if (trans_x) std::swap(lhs_rows, lhs_cols);
    if (trans_y) std::swap(rhs_rows, rhs_cols);

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);

    // TODO: Add Mempool support

    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), output->flat<T>());
      return OkStatus();
    }
    auto out_reshaped = output->template flat_inner_dims<T, 3>();

    std::vector<int> m_array(batch_size, M);
    std::vector<int> n_array(batch_size, N);
    std::vector<int> k_array(batch_size, K);
    std::vector<int> lda_array(batch_size, trans_x ? M : K);
    std::vector<int> ldb_array(batch_size, trans_y ? K : N);
    std::vector<int> ldc_array(batch_size, N);
    std::vector<float> alpha_array(batch_size, 1.0);
    std::vector<float> beta_array(batch_size, 0.0);
    std::vector<int> group_size(1, batch_size);
    std::vector<const T *> a_array;
    std::vector<const T *> b_array;
    std::vector<T *> c_array;
    std::vector<const T *> add_array;

    a_array.reserve(batch_size);
    b_array.reserve(batch_size);
    c_array.reserve(batch_size);
    add_array.reserve(output->dim_size(0));

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
    bool cblasRowMajor = 1;

    zenBatchMatMul(cblasRowMajor, trans_x, trans_y, &m_array[0], &n_array[0],
                   &k_array[0], &alpha_array[0], &a_array[0], &lda_array[0],
                   &b_array[0], &ldb_array[0], &beta_array[0], &c_array[0],
                   &ldc_array[0], 1, &group_size[0], 0, &add_array[0], mul_node,
                   output->dim_size(0));

    Tensor output_reshaped;
    if (output->dims() != 3) {
      TF_RETURN_IF_ERROR(EinsumHelper::ReshapeToRank3(
          *output, bcast.output_batch_size(), &output_reshaped));
    }

    return OkStatus();
  }
};

template <typename Device, typename T>
class ZenEinsum : public OpKernel {
 public:
  explicit ZenEinsum(OpKernelConstruction *c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("equation", &zen_equation_));
    OP_REQUIRES_OK(c, ParseEinsumEquation(
                          zen_equation_, &zen_input_labels_,
                          &zen_output_labels_, &zen_label_types_,
                          &zen_input_label_counts_, &zen_output_label_counts_,
                          &zen_input_has_ellipsis_, &zen_output_has_ellipsis_));
    OP_REQUIRES_OK(c, c->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(c, c->GetAttr("reset", &reset));
  }

  virtual ~ZenEinsum() {}

  void Compute(OpKernelContext *ctx) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    OperandLabels input_labels(zen_input_labels_);
    Labels output_labels(zen_output_labels_);
    std::vector<EinsumDimensionType> label_types(zen_label_types_);
    OperandLabelCounts input_label_counts(zen_input_label_counts_);
    LabelCounts output_label_counts(zen_output_label_counts_);
    LabelToDimSizes label_to_dim_sizes;

    OP_REQUIRES_OK(ctx, EinsumHelper::ProcessDimensions(
                            inputs, zen_input_has_ellipsis_,
                            zen_output_has_ellipsis_, &input_labels,
                            &output_labels, &label_types, &input_label_counts,
                            &output_label_counts, &label_to_dim_sizes));
    // The reduction phase (a) sums across reduction dimensions, (b) takes
    // generalized diagonals, and (c) reshapes it into shape
    //   [(broadcasting) batch shape] + [F,C]
    // where F and C denote the total (compacted) size of free and contract
    // dimensions, respectively.
    const int num_inputs = inputs.size();
    OperandLabels free_labels(num_inputs);
    gtl::InlinedVector<Tensor, 2> inputs_reduced(num_inputs);
    gtl::InlinedVector<bool, 2> swap_free_and_contract(num_inputs);

    for (int i = 0; i < num_inputs; ++i) {
      OP_REQUIRES_OK(ctx,
                     EinsumHelper::ReduceOperand<Device, T>(
                         ctx, inputs[i], label_types, input_label_counts[i],
                         &input_labels[i], &free_labels[i],
                         &swap_free_and_contract[i], &inputs_reduced[i]));
    }
    // After reduction, the inputs should be reshaped to Tensors suitable for
    // contraction. If num_inputs is 1, the reduced input is simply forwarded to
    // the output.
    Tensor contraction_output_reshaped;
    OP_REQUIRES_OK(ctx, ZenEinsumHelper::ZenContractOperands<Device, T>(
                            ctx, inputs_reduced, swap_free_and_contract,
                            &contraction_output_reshaped, out_links, reset));
    // Copy the batch labels from the contraction output. Recover the batch
    // shape, which may have been broadcasted.
    TensorShape result_shape = contraction_output_reshaped.shape();
    result_shape.RemoveLastDims(2);
    int num_labels = label_types.size();
    Labels result_labels;
    // All batch dimensions should be present in the contracted result. First
    // the broadcasting dimensions, then the named batch dimensions.
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBroadcasting)
        result_labels.push_back(label);
    }
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBatch)
        result_labels.push_back(label);
    }
    for (int i = 0; i < num_inputs; ++i) {
      for (int label : free_labels[i]) {
        result_labels.push_back(label);
        result_shape.AddDim(label_to_dim_sizes[label]);
      }
    }
    // Reshape the contraction (or reduction) result to its expanded shape:
    // [(broadcasted) batch shape] + [free shape 0] + [free shape 1].
    Tensor contraction_output;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::CopyFrom(contraction_output_reshaped, result_shape,
                                    &contraction_output));
    // Inflate the output if necessary. (E.g. for the equation 'i->iii' which
    // may arise while computing gradient of a regular Einsum).
    Tensor output_inflated;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::StrideOrInflate<Device, T>(
                 ctx, contraction_output, result_labels, output_label_counts,
                 true /* should_inflate */, &output_inflated));
    if (output_inflated.dims() > contraction_output.dims()) {
      // We inflated the output. Modify result labels accordingly.
      Labels inflated_labels;
      for (int label : result_labels) {
        inflated_labels.insert(inflated_labels.end(),
                               output_label_counts[label], label);
      }
      result_labels.swap(inflated_labels);
    }
    // Find the permutation to map the result labels to the output labels. Note
    // that both the result and the final output may have the repeated labels,
    // in which case the permutation preserves the left-to-right ordering.
    // E.g. if result labels are [0, 0, 1] and output is [0, l, 0] then the
    // permutation should be [0, 2, 1]. We also use the fact that repeated
    // labels in the result are adjacent to each other.
    std::vector<int> output_permutation(output_labels.size());
    std::vector<int> label_to_position(num_labels, -1);
    for (int i = 0; i < result_labels.size(); ++i) {
      // Remember the position of only the leftmost result label.
      if (label_to_position[result_labels[i]] == -1) {
        label_to_position[result_labels[i]] = i;
      }
    }
    for (int i = 0; i < output_labels.size(); ++i) {
      output_permutation[i] = label_to_position[output_labels[i]];
      // We have found the leftmost occurrence. The next one would be adjacent.
      label_to_position[output_labels[i]] += 1;
    }

    Tensor output;
    OP_REQUIRES_OK(ctx, EinsumHelper::TransposeOperand<Device, T>(
                            ctx, output_inflated, output_permutation, &output));
    ctx->set_output(0, output);
  }

 private:
  string zen_equation_;
  OperandLabels zen_input_labels_;
  Labels zen_output_labels_;
  std::vector<EinsumDimensionType> zen_label_types_;
  OperandLabelCounts zen_input_label_counts_;
  LabelCounts zen_output_label_counts_;
  gtl::InlinedVector<bool, 2> zen_input_has_ellipsis_;
  bool zen_output_has_ellipsis_ = false;
  int out_links;
  bool reset;
};

#define REGISTER_EINSUM_ZEN(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ZenEinsum").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenEinsum<CPUDevice, TYPE>);

TF_CALL_float(REGISTER_EINSUM_ZEN);
// TF_CALL_bfloat16(REGISTER_EINSUM_ZEN);
}  // namespace tensorflow
