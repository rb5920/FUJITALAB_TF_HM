cc_library(
    name = "TLibVideoIO",
    srcs = ["TVideoIOYuv.cpp"],
    hdrs = ["TVideoIOYuv.h"],	
	copts = ["-Itensorflow/HM_BAZEL"],
	deps = ["//tensorflow/HM_BAZEL/TLibCommon:TLibCommon_lb"],
    visibility = ["//visibility:public"],
)
cc_inc_library(
    name = "TLibVideoIO_lb",
    hdrs = ["TVideoIOYuv.h"],
	visibility = ["//visibility:public"],
)
