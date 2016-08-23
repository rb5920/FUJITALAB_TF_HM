all: build

BAZEL=$(shell which bazel)

.PHONY: build
build : 
		$(BAZEL) build //tensorflow/HM_BAZEL/NEURAL_CUP:NEURAL_CUP
		$(BAZEL) build //tensorflow/HM_BAZEL/TLibVideoIO:TLibVideoIO
		$(BAZEL) build //tensorflow/HM_BAZEL/libmd5:libmd5
		$(BAZEL) build //tensorflow/HM_BAZEL/TLibCommon:TLibCommon
		$(BAZEL) build //tensorflow/HM_BAZEL/TLibDecoder:TLibDecoder
		$(BAZEL) build //tensorflow/HM_BAZEL/TLibEncoder:TLibEncoder
		$(BAZEL) build //tensorflow/HM_BAZEL/TAppCommon:TAppCommon
		$(BAZEL) build //tensorflow/HM_BAZEL/TAppDecoder:TAppDecoder
		$(BAZEL) build //tensorflow/HM_BAZEL/TAppEncoder:TAppEncoder

.PHONY: clean
clean :
		$(BAZEL) clean

