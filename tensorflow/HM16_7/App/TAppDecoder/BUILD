
cc_binary(
    name = "TAppDecoder",
    srcs = ["decmain.cpp"],
	copts = ["-Itensorflow/HM16_7/Lib"],
	deps = [":TAppDecoder_lib"]
)
cc_library(
    name = "TAppDecoder_lib",
    srcs = ["TAppDecCfg.cpp",
			"TAppDecTop.cpp"],
    hdrs = ["TAppDecCfg.h",
			"TAppDecTop.h"],
	copts = ["-Itensorflow/HM16_7/Lib"],
	deps = ["//tensorflow/HM16_7/Lib/TLibDecoder:TLibDecoder",
			"//tensorflow/HM16_7/Lib/TAppCommon:TAppCommon"],
)
