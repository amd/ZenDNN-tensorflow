# NEW
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

# NEW
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)

all_link_actions = [
    # NEW
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "/usr/lib/llvm-14/bin/clang",
        ),
        tool_path(
            name = "ld",
            path = "/usr/lib/llvm-14/bin/ld64.lld",
        ),
        tool_path(
            name = "ar",
            path = "/usr/lib/llvm-14/bin/llvm-libtool-darwin",
        ),
        tool_path(
            name = "cpp",
            path = "/usr/lib/llvm-14/bin/clang++",
        ),
        tool_path(
            name = "gcov",
            path = "/usr/lib/llvm-14/bin/llvm-cov",
        ),
        tool_path(
            name = "nm",
            path = "/usr/lib/llvm-14/bin/llvm-nm",
        ),
        tool_path(
            name = "objdump",
            path = "/usr/lib/llvm-14/bin/llvm-objdump",
        ),
        tool_path(
            name = "strip",
            path = "/usr/lib/llvm-14/bin/llvm-strip",
        ),
    ]

    features = [
        # NEW
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "--target=x86_64-apple-darwin",
                                "-lSystem",
                                "-fuse-ld=/usr/lib/llvm-14/bin/ld64.lld"
                            ],
                        ),
                    ]),
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,  # NEW
        cxx_builtin_include_directories = [
            "%sysroot%/usr/include",
            "/usr/lib/llvm-14/lib/clang/14.0.6/include",
            "%sysroot%/System/Library/Frameworks/",

        ],
        toolchain_identifier = "linux-to-mac-toolchain",
        host_system_name = "linux",
        target_system_name = "macos",
        target_cpu = "x86_64",
        target_libc = "macosx",
        compiler = "clang",
        abi_version = "",
        abi_libc_version = "",
        tool_paths = tool_paths,
        builtin_sysroot = "/usr/local/google/home/srnitin/mac_sdk/MacOSX.sdk",
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)

