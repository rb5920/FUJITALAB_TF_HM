
cc_binary(
    name = "TAppDecoder",
    srcs = ["decmain.cpp"],
	copts = ["-Itensorflow/HM_BAZEL"],
	deps = [":TAppDecoder_lib"]
)
cc_library(
    name = "TAppDecoder_lib",
    srcs = ["TAppDecCfg.cpp",
			"TAppDecTop.cpp"],
    hdrs = ["TAppDecCfg.h",
			"TAppDecTop.h"],
	copts = ["-Itensorflow/HM_BAZEL"],
	deps = ["//tensorflow/HM_BAZEL/TLibDecoder:TLibDecoder",
			"//tensorflow/HM_BAZEL/TAppCommon:TAppCommon",],
)
