# Description:
#   Tensorflow C++ inference example for labeling images.

visibility = ["//visibility:public"],

#licenses(["notice"])  # Apache 2.0

#exports_files(["LICENSE"])


cc_library(
    name = "NEURAL_CUP",
    srcs = ["neural_cup.cc"],
	hdrs = ["neural_cup.h"],
	copts = ["-Itensorflow/HM_BAZEL/NEURAL_CUP"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
    ],
	visibility = ["//visibility:public"],
)


