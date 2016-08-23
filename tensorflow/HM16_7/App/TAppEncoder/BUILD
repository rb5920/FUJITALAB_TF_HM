cc_binary(
    name = "TAppEncoder",
    srcs = ["encmain.cpp"],
	copts = ["-Itensorflow/HM16_7/Lib"],
	deps = [":TAppEncoder_lib"]
)
cc_library(
    name = "TAppEncoder_lib",
    srcs = ["TAppEncCfg.cpp",
			"TAppEncTop.cpp"],
    hdrs = ["TAppEncCfg.h",
			"TAppEncTop.h"],
	copts = ["-Itensorflow/HM16_7/Lib"],
	deps = ["//tensorflow/HM16_7/Lib/TLibEncoder:TLibEncoder",
			"//tensorflow/HM16_7/Lib/TAppCommon:TAppCommon",],
)
