genrule(
    name="blis_build",
    srcs = [
        "configure",
        "build/config.mk.in",
        "build/bli_config.h.in",
        "Makefile",
        "common.mk",
    ],
    cmd = "mkdir -p $(RULEDIR)/include/zen/ " +
          "&& cp --verbose -- external/amd_blis/bli_config.h $(location include/zen/bli_config.h) " +
          "&& cp --verbose -- external/amd_blis/include/**/blis.h $(location include/zen/blis.h) " +
          "&& cp --verbose -- external/amd_blis/include/**/cblas.h $(location include/zen/cblas.h) " +
          "&& cp --verbose -- external/amd_blis/lib/**/libblis-mt.so $(location  libblis-mt.so) " +
          "&& cp --verbose -- external/amd_blis/lib/**/libblis-mt.so.4 $(location  libblis-mt.so.4) ",
    outs = [
        "include/zen/bli_config.h",
        "include/zen/blis.h",
        "include/zen/cblas.h",
        "libblis-mt.so",
        "libblis-mt.so.4",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "amd_blis",
    srcs = [ ":blis_build", ],
    includes = [ "include/amdzen/", "external/amd_blis/"],
    visibility = ["//visibility:public"],
)
