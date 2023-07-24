exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party/zen_dnn:build_defs.bzl",
    "if_zendnn",
)

_COPTS_LIST = select({
    "//conditions:default": ["-fexceptions -fopenmp -march=znver2"],
}) + [
    "-DBIAS_ENABLED=1",
    "-DZENDNN_ENABLE=1",
    "-DLIBM_ENABLE=1",
    "-DZENDNN_X64=1",
] + ["-Iexternal/amd_blis/include/zen/"]

_INCLUDES_LIST = [
    "inc",
    "include",
    "src",
    "src/common",
    "src/common/ittnotify",
    "src/cpu",
    "src/cpu/gemm",
    "src/cpu/x64/xbyak",
]

_TEXTUAL_HDRS_LIST = glob([
    "inc/*",
    "include/**/*",
    "src/common/*.hpp",
    "src/common/ittnotify/**/*.h",
    "src/cpu/*.hpp",
    "src/cpu/**/*.hpp",
    "src/cpu/jit_utils/**/*.hpp",
    "src/cpu/x64/xbyak/*.h",
])

# Large autogen files take too long time to compile with usual optimization
# flags. These files just generate binary kernels and are not the hot spots,
# so we factor them out to lower compiler optimizations in ":dnnl_autogen".
#cc_library(
#    name = "zendnn_autogen",
#    srcs = glob(["src/cpu/x64/gemm/**/*_kern_autogen*.cpp"]),
#    copts = select({
#        "@org_tensorflow//tensorflow:macos": ["-O0"],
#        "//conditions:default": ["-O1"],
#    }) + ["-U_FORTIFY_SOURCE"] + _COPTS_LIST,
#    includes = _INCLUDES_LIST,
#    textual_hdrs = _TEXTUAL_HDRS_LIST,
#    visibility = ["//visibility:public"],
#)

cc_library(
    name = "zen_dnn",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/*.cpp",
            "src/cpu/**/*.cpp",
        ],
        exclude = [
            "src/cpu/aarch64/**",
            "src/common/ittnotify/*.c",
#            "src/cpu/x64/gemm/**/*_kern_autogen.cpp",
        ],
    ) + [
          "@amd_libm//:libm_config",
    ],
    copts = _COPTS_LIST,
    includes = _INCLUDES_LIST,
    # TODO(penpornk): Use lrt_if_needed from tensorflow.bzl instead.
    linkopts = select({
        "@org_tensorflow//tensorflow:linux_aarch64": ["-lrt"],
        "@org_tensorflow//tensorflow:linux_x86_64": ["-lrt"],
        "@org_tensorflow//tensorflow:linux_ppc64le": ["-lrt"],
        "//conditions:default": [],
    }),
    textual_hdrs = _TEXTUAL_HDRS_LIST,
    visibility = ["//visibility:public"],
    deps = [ "@amd_blis//:amd_blis" ] + if_zendnn(
        ["@org_tensorflow//third_party/zen_dnn:amd_binary_blob"],
    ),
)
