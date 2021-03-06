all: build

BAZEL=$(shell which bazel)

.PHONY: buildHM
buildHM : 
#		$(BAZEL) build //tensorflow/HM16_7/Lib/NEURAL_CUP:NEURAL_CUP
		$(BAZEL) build //tensorflow/HM16_7/Lib/TLibVideoIO:TLibVideoIO
		$(BAZEL) build //tensorflow/HM16_7/Lib/libmd5:libmd5
		$(BAZEL) build //tensorflow/HM16_7/Lib/TLibCommon:TLibCommon
		$(BAZEL) build //tensorflow/HM16_7/Lib/TLibDecoder:TLibDecoder
		$(BAZEL) build //tensorflow/HM16_7/Lib/TLibEncoder:TLibEncoder
		$(BAZEL) build //tensorflow/HM16_7/Lib/TAppCommon:TAppCommon
		$(BAZEL) build //tensorflow/HM16_7/App/TAppDecoder:TAppDecoder
		$(BAZEL) build //tensorflow/HM16_7/App/TAppEncoder:TAppEncoder

.PHONY: clean
clean :
		$(BAZEL) clean

