load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "decode_detections",
    srcs = ["lib/decode_detections.cpp"],
    hdrs = ["lib/decode_detections.hpp"],
    copts = ["-O3","-fPIC","-DNDEBUG"],
    deps = ["@eigen"],
)

cc_binary(
    name = "inference_dataset",
    srcs = ["src/inference_dataset.cpp"],
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
	":decode_detections",
    ],
    copts = ["-O3","-fPIC","-DNDEBUG"],
)


