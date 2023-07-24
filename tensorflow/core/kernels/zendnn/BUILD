load(
    "//tensorflow:tensorflow.bzl",
    "tf_zendnn_kernel_library",
)
load(
    "//third_party/zen_dnn:build_defs.bzl",
    "zendnn_deps",
)

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

MATH_DEPS = [
    "//tensorflow/core/kernels:fill_functor",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:math_grad",
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core/framework:op_requires",
    "//third_party/eigen3",
]

ARRAY_DEPS = [
    "//tensorflow/core/kernels:concat_lib",
    "//tensorflow/core/kernels:fill_functor",
    "//tensorflow/core/kernels:gather_functor",
    "//tensorflow/core/kernels:ops_util",
    "//tensorflow/core/kernels:transpose_functor",
    "//tensorflow/core:array_grad",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/framework:bounds_check",
    "//third_party/eigen3",
]

NN_DEPS = [
    "//tensorflow/core/kernels:ops_util",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:nn_grad",
    "//tensorflow/core/framework:bounds_check",
    "//third_party/eigen3",
]

tf_zendnn_kernel_library(
    name = "zen_matmul_op",
    srcs = [
        "zen_matmul_op.cc",
        "zen_batch_matmul_op.cc",
        "zen_matmul_op_fused.cc",
        ],
    hdrs = [
            "zen_matmul_ops_util.h",
            ],
    deps = MATH_DEPS + [
        "//tensorflow/core/kernels:fused_eigen_output_kernels",
    ] + zendnn_deps(),
)

tf_zendnn_kernel_library(
    name = "zen_conv_ops",
    srcs = [
        "zen_conv_ops.cc",
        "zen_conv_ops_fused.cc",
        "zen_conv_sum_ops_fused.cc",
        "zen_inception_ops.cc",
        "zen_op_utils.cc",
        ],
    hdrs = [
            "zen_conv_ops_util.h",
            ],
    deps = [
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:conv_ops",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core:framework",
        ] + zendnn_deps(),
)

tf_zendnn_kernel_library(
    name = "zen_pool_ops",
    srcs = [
            "zen_max_pool_ops.cc",
            "zen_avg_pool_ops.cc",
            ],
    hdrs = [],
    deps = [
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels:pooling_ops",
       "//tensorflow/core/framework:bounds_check",
       "//tensorflow/core:framework",
       ] + zendnn_deps(),
)

tf_zendnn_kernel_library(
    name = "zen_concat_ops",
    srcs = ["zen_concat_op.cc"],
    hdrs = [],
    deps = [
        "//tensorflow/core/kernels:concat_lib_hdrs",
        "//tensorflow/core/kernels:pooling_ops",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
    ] + zendnn_deps(),
)

tf_zendnn_kernel_library(
    name = "zen_cwise_op",
    srcs = [
        "zen_cwise_ops_common.cc",
        ],
    hdrs = [
            "zen_cwise_ops_common.h",
            ],
    deps = MATH_DEPS + [
	"//tensorflow/core/kernels:cwise_op",
    ] + zendnn_deps(),
)

tf_zendnn_kernel_library(
    name = "zen_reshape_op",
    srcs = [
        "zen_reshape_op.cc",
    ],
    hdrs = [],
    deps = ARRAY_DEPS + zendnn_deps(),
)

tf_zendnn_kernel_library(
    name = "zen_transpose_op",
    srcs = [
        "zen_transpose_op.cc",
    ],
    hdrs = ["zen_transpose_op.h"],
    deps = ARRAY_DEPS + zendnn_deps(),
)

tf_zendnn_kernel_library(
    name = "zen_softmax_op",
    srcs = ["zen_softmax_op.cc"],
    hdrs = [],
    deps = [
            "//tensorflow/core/framework:bounds_check",
            "//tensorflow/core/kernels:ops_util",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
           ] + zendnn_deps(),
)


tf_zendnn_kernel_library(
    name = "zen_fused_batch_norm_op",
    srcs = ["zen_fused_batch_norm_op.cc"],
    hdrs = [],
    deps = NN_DEPS + [
            "//tensorflow/core/kernels:fill_functor",
            "//tensorflow/core/kernels:redux_functor",
            "//tensorflow/core/kernels:transpose_functor",
            "//tensorflow/core/kernels:fused_batch_norm_op",
            "//tensorflow/core/kernels:no_op",
           ] + zendnn_deps(),
)
