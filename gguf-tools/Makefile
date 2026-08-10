CC ?= cc
CXX ?= c++
UNAME_S := $(shell uname -s)
LLAMA_CPP_DIR ?= ../../deepseek-v4-quants/llama.cpp
LLAMA_CPP_BUILD ?= $(LLAMA_CPP_DIR)/build-remote
LLAMA_CPP_BIN ?= $(LLAMA_CPP_BUILD)/bin
LLAMA_CPP_CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 \
	-I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
QUALITY_LDLIBS := -lm -pthread -framework Foundation -framework Metal
QUALITY_TARGETS := ds4.o lgn.o ds4_metal.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o rax.o
QUALITY_OBJS := ../ds4.o ../lgn.o ../ds4_metal.o ../ds4_distributed.o ../ds4_tp.o ../ds4_ssd.o ../ds4_layer_pack.o ../rax.o
QUALITY_LINK := $(CC) $(CFLAGS) -I.. -o
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -lggml-blas -lggml-metal \
	-framework Foundation -framework Metal -framework Accelerate
else
NATIVE_CPU_FLAG ?= -march=native
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_BIN) -Wl,-rpath,$(abspath $(LLAMA_CPP_BIN)) \
	-lllama -lggml -lggml-base -lggml-cpu -ldl -lpthread
endif

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quality-score quality-llama-score

all: deepseek4-quantize

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

ifeq ($(UNAME_S),Darwin)
quality-score:
	$(MAKE) -C .. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.c $(QUALITY_OBJS) $(QUALITY_LDLIBS)
else
quality-score:
	@echo "error: gguf-tools quality-score requires Darwin/Apple Metal (got $(UNAME_S))" >&2
	@exit 2
endif

quality-llama-score:
	$(CXX) $(LLAMA_CPP_CXXFLAGS) -o quality-testing/score_llama quality-testing/score_llama.cpp $(LLAMA_CPP_LDLIBS)

clean:
	rm -f deepseek4-quantize quality-testing/score_official quality-testing/score_llama
