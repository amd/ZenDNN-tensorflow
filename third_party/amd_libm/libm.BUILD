genrule(
    name="libm_config",
    cmd = "cd external/amd_libm" +
          "&& scons -j32" +
          "&& cd ../.." +
          "&& cp --verbose -- external/amd_libm/build/aocl-release/src/libalm.so $(location libalm.so) ",
    outs = [
        "libalm.so",
],
    visibility = ["//visibility:public"],
)
