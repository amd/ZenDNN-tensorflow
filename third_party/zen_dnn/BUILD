load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

config_setting(
    name = "build_with_zendnn",
    define_values = {
        "build_with_zendnn": "true",
        "build_with_openmp": "true",
    },
    visibility = ["//visibility:public"],
)

filegroup(
    name = "LICENSES",
    srcs = [
        "LICENSE",
        "@llvm_openmp//:LICENSE.txt",
    ],
    visibility = ["//visibility:public"],
)

# TODO(AMD-tf) Remove the following 3 calls to cc_library and replace all uses
# of zen_libs_* with @llvm_openmp//:libiomp5.* directly.

cc_library(
    name = "zen_libs_linux",
    srcs = [
        "@llvm_openmp//:libiomp5.so",
        "@amd_blis//:libblis-mt.so",
        "@amd_blis//:libblis-mt.so.4",
    ],
    visibility = ["//visibility:public"],
)

# MacOS build configuration is provided for completness, it has not been tested
cc_library(
    name = "zen_libs_darwin",
    srcs = [
        "@llvm_openmp//:libiomp5.dylib",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "zen_libs_windows",
    srcs = [
        "@llvm_openmp//:libiomp5md.dll",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "amd_binary_blob",
    visibility = ["//visibility:public"],
    deps = select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            ":zen_libs_linux",
        ],
        "@org_tensorflow//tensorflow:macos": [
            ":zen_libs_darwin",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "zen_libs_windows",
        ],
        "//conditions:default": [],
    }),
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)
