load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tf2xla/api/v1:__subpackages__",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:__subpackages__",
    ],
)

cc_library(
    name = "compilation_timer",
    hdrs = ["compilation_timer.h"],
    deps = [
        "//tensorflow/core/platform:profile_utils_cpu_utils",
    ],
)

tf_cc_test(
    name = "compilation_timer_test",
    srcs = ["compilation_timer_test.cc"],
    deps = [
        ":compilation_timer",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_matchers",
    testonly = True,
    hdrs = ["test_matchers.h"],
    deps = [
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "test_matchers_test",
    srcs = ["test_matchers_test.cc"],
    deps = [
        ":test_matchers",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "mlir_pass_instrumentation",
    srcs = ["mlir_pass_instrumentation.cc"],
    hdrs = ["mlir_pass_instrumentation.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "mlir_pass_instrumentation_test",
    srcs = ["mlir_pass_instrumentation_test.cc"],
    deps = [
        ":mlir_pass_instrumentation",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:compile_mlir_util_no_tf_dialect_passes",
        "//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "legalize_tf_mlir",
    srcs = ["legalize_tf_mlir.cc"],
    hdrs = ["legalize_tf_mlir.h"],
    deps = [
        ":compilation_timer",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/mlir/tensorflow/transforms:set_tpu_infeed_layout",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu:tpu_compile",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:error_logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "legalize_tf_to_hlo",
    srcs = ["legalize_tf_to_hlo.cc"],
    hdrs = ["legalize_tf_to_hlo.h"],
    deps = [
        ":compilation_timer",
        ":legalize_tf_mlir",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:compile_tf_graph",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_tpu_backend_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:compile_only_client",
    ],
)

tf_cc_test(
    name = "legalize_tf_mlir_test",
    srcs = ["legalize_tf_mlir_test.cc"],
    deps = [
        ":legalize_tf_mlir",
        ":test_matchers",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
    ],
)

tf_cc_test(
    name = "legalize_tf_to_hlo_test",
    srcs = ["legalize_tf_to_hlo_test.cc"],
    deps = [
        ":legalize_tf_to_hlo",
        ":test_matchers",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/stream_executor",
    ],
)
