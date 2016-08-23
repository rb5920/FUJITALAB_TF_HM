cc_binary(
    name = "TAppEncoder",
    srcs = ["encmain.cpp"],
	copts = ["-Itensorflow/HM_BAZEL"],
	deps = [":TAppEncoder_lib"]
)
cc_library(
    name = "TAppEncoder_lib",
    srcs = ["TAppEncCfg.cpp",
			"TAppEncTop.cpp"],
    hdrs = ["TAppEncCfg.h",
			"TAppEncTop.h"],
	copts = ["-Itensorflow/HM_BAZEL"],
	deps = ["//tensorflow/HM_BAZEL/TLibEncoder:TLibEncoder",
			"//tensorflow/HM_BAZEL/TAppCommon:TAppCommon",],
)
