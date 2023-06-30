package(default_visibility = ["//visibility:public"])
load(":cc_toolchain_config.bzl", "cc_toolchain_config")

cc_toolchain_suite(
    name = "mac_cross_compile",
    toolchains = {
        "darwin_x86_64": ":linux_to_mac",
"darwin": ":linux_to_mac",
    },
)

filegroup(name = "empty")

cc_toolchain(
    name = "linux_to_mac",
    toolchain_identifier = "linux-to-mac-toolchain",
    toolchain_config = ":linux_to_mac_toolchain_config",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
)

cc_toolchain_config(
    name = "linux_to_mac_toolchain_config"
)

platform(
    name = "darwin",
    constraint_values = [
       "@platforms//cpu:x86_64",
       "@platforms//os:macos",
    ],
)
