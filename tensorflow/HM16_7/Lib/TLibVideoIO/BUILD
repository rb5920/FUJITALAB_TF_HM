cc_library(
    name = "TLibVideoIO",
    srcs = ["TVideoIOYuv.cpp"],
    hdrs = ["TVideoIOYuv.h"],	
	copts = ["-Itensorflow/HM16_7/Lib/"],
	deps = ["//tensorflow/HM16_7/Lib/TLibCommon:TLibCommon_lb"],
    visibility = ["//visibility:public"],
)
cc_inc_library(
    name = "TLibVideoIO_lb",
    hdrs = ["TVideoIOYuv.h"],
	visibility = ["//visibility:public"],
)
