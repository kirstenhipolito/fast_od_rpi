load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "decode_detections",
    srcs = ["decode_detections.cpp"],
    hdrs = ["decode_detections.hpp"],
    copts = ["-O3","-fPIC","-DNDEBUG"],
    deps = ["@eigen"],
    visibility = ["//main:__pkg__"],
)
