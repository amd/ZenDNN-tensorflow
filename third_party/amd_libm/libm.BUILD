genrule(
    name="libm_config",
    cmd = "cp --verbose -- external/amd_libm/build/aocl-release/src/libalm.so $(location libalm.so) ",
    outs = [
        "libalm.so",
],
    visibility = ["//visibility:public"],
)
