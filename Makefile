CC ?= cc
UNAME_S := $(shell uname -s)
.DEFAULT_GOAL := all

ifneq ($(UNAME_S),Darwin)
$(error Laguna Metal-only build requires Darwin/Apple Metal (got $(UNAME_S)))
endif

NATIVE_CPU_FLAG ?= -mcpu=native
SAMPLING_TEST := tests/test_sampling
METAL_EXACT_TEST := test-glm-q23-metal

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread
# Metal kernels are loaded from these files at runtime by ds4_metal.m.  Keep
# the loader's environment-name/default-path pairs explicit for the source
# check below, but do not make the files compile-time prerequisites of the
# large Objective-C objects.
METAL_SOURCE_SPECS := \
	DS4_METAL_FLASH_ATTN_SOURCE=metal/flash_attn.metal \
	DS4_METAL_DENSE_SOURCE=metal/dense.metal \
	DS4_METAL_MOE_SOURCE=metal/moe.metal \
	DS4_METAL_DSV4_HC_SOURCE=metal/dsv4_hc.metal \
	DS4_METAL_UNARY_SOURCE=metal/unary.metal \
	DS4_METAL_DSV4_KV_SOURCE=metal/dsv4_kv.metal \
	DS4_METAL_DSV4_ROPE_SOURCE=metal/dsv4_rope.metal \
	DS4_METAL_DSV4_MISC_SOURCE=metal/dsv4_misc.metal \
	DS4_METAL_LAGUNA_SOURCE=metal/laguna.metal \
	DS4_METAL_DFLASH_SOURCE=metal/dflash.metal \
	DS4_METAL_ARGSORT_SOURCE=metal/argsort.metal \
	DS4_METAL_CPY_SOURCE=metal/cpy.metal \
	DS4_METAL_CONCAT_SOURCE=metal/concat.metal \
	DS4_METAL_GET_ROWS_SOURCE=metal/get_rows.metal \
	DS4_METAL_SUM_ROWS_SOURCE=metal/sum_rows.metal \
	DS4_METAL_SOFTMAX_SOURCE=metal/softmax.metal \
	DS4_METAL_REPEAT_SOURCE=metal/repeat.metal \
	DS4_METAL_GLU_SOURCE=metal/glu.metal \
	DS4_METAL_NORM_SOURCE=metal/norm.metal \
	DS4_METAL_BIN_SOURCE=metal/bin.metal \
	DS4_METAL_SET_ROWS_SOURCE=metal/set_rows.metal
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_DFLASH ?=
# Deliberately empty: the model-backed Laguna integration gate must never
# pretend that the legacy DS4_TEST_MODEL default is a supported fixture.
LAGUNA_TEST_MODEL ?=

METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o lgn.o lgn_model.o lgn_dflash.o lgn_graph.o lgn_dflash_graph.o lgn_dflash_exec.o ds4_ssd.o ds4_metal.o

DS4_TEST_METAL_OBJ := ds4_metal_test_hooks.o
DS4_TEST_DS4_OBJ := ds4_test_hooks.o
TEST_CORE_OBJS := $(filter-out ds4.o ds4_metal.o,$(CORE_OBJS)) $(DS4_TEST_DS4_OBJ) $(DS4_TEST_METAL_OBJ)

METAL_SOURCE_ORDER_ONLY := | check-metal-sources

UNSUPPORTED_TARGETS := cpu cuda cuda-spark cuda-generic cuda-regression strix-halo rocm \
	test-mxfp4-cuda test-cuda-session-batch test-cuda-mixed-batch

.PHONY: all help clean test test-legacy test-engine-lifecycle test-metal-laguna test-metal-laguna-integration test-laguna-cli-options test-lgn check-metal-sources test-metal-session-batch test-mxfp4-metal test-glm-q23-metal dflash-verify-depth $(UNSUPPORTED_TARGETS)

# Keep this check cheap and always current: the executable contains only the
# host-side loader, while these source files are read and compiled at runtime.
# A fixed pair list means a missing default cannot disappear from a wildcard;
# an explicitly nonempty override is authoritative and the default is not
# consulted, matching the fail-closed runtime override contract.
check-metal-sources:
	@set -eu; for spec in $(METAL_SOURCE_SPECS); do \
		env_name=$${spec%%=*}; \
		default_path=$${spec#*=}; \
		override_path=$$(printenv "$$env_name" 2>/dev/null || true); \
		if test -n "$$override_path"; then \
			test -f "$$override_path" && test -r "$$override_path" || { echo "error: unreadable Metal source override $$env_name=$$override_path" >&2; exit 1; }; \
		else \
			test -f "$$default_path" && test -r "$$default_path" || { echo "error: missing runtime Metal source $$default_path (set $$env_name to override)" >&2; exit 1; }; \
		fi; \
	done

.PHONY: metal-decode-schedule-bench metal-prefill-variant-bench check-mxfp4-half-lut

all: check-metal-sources ds4 ds4-server ds4-bench ds4-eval

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, and ./ds4-eval"
	@echo "  make test         Build/run the model-independent Apple Metal/Laguna suite"
	@echo "  make test-metal-laguna  Run the strict model-independent Apple Metal/Laguna suite"
	@echo "  make test-legacy  Run the temporary umbrella regression suite (may need a model)"
	@echo "  make test-metal-laguna-integration LAGUNA_TEST_MODEL=FILE  Run model-backed Laguna smoke"
	@echo "  make metal-decode-schedule-bench  Build the balanced Metal decode schedule benchmark"
	@echo "  make metal-prefill-variant-bench  Build the balanced Metal prefill variant benchmark"
	@echo "  make check-mxfp4-half-lut  Verify the checked-in MXFP4 half LUT matches the generator"
	@echo "  make test-mxfp4-metal  Check the MXFP4 half LUT, then run Metal MXFP4 exactness tests"
	@echo "  make dflash-verify-depth  Run DFlash speculative verification smoke if support GGUF is present"
	@echo "  CPU/CUDA/ROCm targets are unsupported in this Darwin/Apple Metal-only fork"
	@echo "  make clean        Remove build outputs"

$(UNSUPPORTED_TARGETS):
	@echo "error: make $@ is unsupported; this build requires Darwin/Apple Metal" >&2
	@exit 2

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

quality/score_official: quality/score_official.c ds4.h $(CORE_OBJS) rax.o | check-metal-sources
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ quality/score_official.c $(CORE_OBJS) rax.o $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_metal_session_batch

speed-bench/metal_decode_schedule_bench.o: speed-bench/metal_decode_schedule_bench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_decode_schedule_bench: speed-bench/metal_decode_schedule_bench.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-decode-schedule-bench: speed-bench/metal_decode_schedule_bench

speed-bench/metal_prefill_variant_bench.o: speed-bench/metal_prefill_variant_bench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_prefill_variant_bench: speed-bench/metal_prefill_variant_bench.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-prefill-variant-bench: speed-bench/metal_prefill_variant_bench

tests/test_mxfp4_metal.o: tests/test_mxfp4_metal.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_mxfp4_metal: tests/test_mxfp4_metal.o ds4_metal.o | check-metal-sources
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

check-mxfp4-half-lut:
	python3 metal/generate_mxfp4_half_lut.py --check

test-mxfp4-metal: check-mxfp4-half-lut tests/test_mxfp4_metal
	./tests/test_mxfp4_metal

tests/test_glm_q23_metal.o: tests/test_glm_q23_metal.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_glm_q23_metal: tests/test_glm_q23_metal.o ds4_metal.o | check-metal-sources
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-glm-q23-metal: tests/test_glm_q23_metal
	./tests/test_glm_q23_metal

tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cpu_test_hooks.o lgn.o lgn_model.o lgn_dflash.o ds4_kvstore.o rax.o ds4_ssd.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_lgn.o: tests/test_lgn.c lgn.h lgn_model.h lgn_dflash.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_lgn: tests/test_lgn.o lgn.o lgn_model.o lgn_dflash.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-lgn: tests/test_lgn
	./tests/test_lgn

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_gpu.h lgn.h lgn_model.h lgn_dflash.h lgn_graph.h lgn_dflash_graph.h lgn_dflash_exec.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

lgn.o: lgn.c lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn.c

lgn_model.o: lgn_model.c lgn_model.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_model.c

lgn_dflash.o: lgn_dflash.c lgn_dflash.h lgn_model.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_dflash.c

lgn_graph.o: lgn_graph.c lgn_graph.h ds4_gpu.h lgn_model.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_graph.c

lgn_dflash_graph.o: lgn_dflash_graph.c lgn_dflash_graph.h lgn_dflash.h lgn_model.h ds4_gpu.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_dflash_graph.c

lgn_dflash_exec.o: lgn_dflash_exec.c lgn_dflash_exec.h lgn_dflash_graph.h lgn_dflash.h lgn_model.h ds4_gpu.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_dflash_exec.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_gpu.h ds4_ssd.h ds4_help.h ds4_kvstore.h rax.h lgn.h lgn_graph.h lgn_dflash_graph.h lgn_dflash_exec.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -c -o $@ tests/ds4_test.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_metal.o: ds4_metal.m ds4_gpu.h | check-metal-sources
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_metal_test_hooks.o: ds4_metal.m ds4_gpu.h | check-metal-sources
	$(CC) $(OBJCFLAGS) -DDS4_TEST_HOOKS -c -o $@ ds4_metal.m

ds4_test_hooks.o: ds4.c ds4.h ds4_ssd.h ds4_gpu.h lgn.h lgn_model.h lgn_dflash.h lgn_graph.h lgn_dflash_graph.h lgn_dflash_exec.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -c -o $@ ds4.c

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_gpu.h lgn.h lgn_model.h lgn_dflash.h lgn_graph.h lgn_dflash_graph.h lgn_dflash_exec.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(TEST_CORE_OBJS) $(METAL_SOURCE_ORDER_ONLY)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(TEST_CORE_OBJS) $(METAL_LDLIBS)

test-engine-lifecycle: ds4_test
	./ds4_test --engine-lifecycle

test-legacy: test-lgn ds4-eval q4k-dot-test mxfp4-dot-test \
	$(SAMPLING_TEST) $(METAL_EXACT_TEST) ds4 ds4-server ds4-bench test-engine-lifecycle
	./ds4-eval --self-test-extractors
	./tests/test_sampling

test-laguna-cli-options: ds4 ds4-server ds4-bench ds4-eval tests/test_laguna_cli_options.sh
	./tests/test_laguna_cli_options.sh

test-metal-laguna: check-metal-sources test-lgn test-glm-q23-metal test-laguna-cli-options test-engine-lifecycle $(SAMPLING_TEST) ds4 ds4-server ds4-bench ds4-eval
	@set -eu; \
	./ds4_test --laguna-architecture --laguna-selector-parser --laguna-session-routes --laguna-graph-lifecycle --laguna-dflash-graph-lifecycle --laguna-dflash-exec --laguna-dflash-command-ownership --server; \
	./ds4_test --dflash-payload-lifecycle; \
	./tests/test_sampling; \
	DS4_TEST_LAGUNA_STAGED_SWA_ALLOW_FALLBACK= \
	./ds4_test --laguna-metal-core

test: test-metal-laguna

test-metal-laguna-integration: check-metal-sources ds4 ds4-server ds4-bench ds4-eval tests/test_metal_session_batch
	@test -n "$(strip $(LAGUNA_TEST_MODEL))" || { \
		echo "error: set LAGUNA_TEST_MODEL=/path/to/laguna-s2.1.gguf for model-backed integration" >&2; \
		exit 2; \
	}
	@test -f "$(LAGUNA_TEST_MODEL)" || { \
		echo "error: Laguna integration model not found: $(LAGUNA_TEST_MODEL)" >&2; \
		exit 2; \
	}
	DS4_TEST_MODEL="$(LAGUNA_TEST_MODEL)" ./ds4 --metal --model "$(LAGUNA_TEST_MODEL)" --inspect
	DS4_TEST_MODEL="$(LAGUNA_TEST_MODEL)" \
	DS4_TEST_BACKEND=metal \
	DS4_TEST_SSD_STREAMING= \
	DS4_TEST_SSD_STREAMING_COLD= \
	DS4_TEST_SSD_STREAMING_CACHE_GB= \
	DS4_TEST_SSD_STREAMING_CACHE_EXPERTS= \
	DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS= \
	./tests/test_metal_session_batch

dflash-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dflash-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ -z "$(strip $(DS4_TEST_DFLASH))" ]; then \
		echo "dflash-verify-depth: skipped, set DS4_TEST_DFLASH=FILE to a DFlash support GGUF"; \
	elif [ ! -f "$(DS4_TEST_DFLASH)" ]; then \
		echo "dflash-verify-depth: skipped, missing DFlash support $(DS4_TEST_DFLASH)"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DFLASH="$(DS4_TEST_DFLASH)" ./ds4_test --dflash-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4_test tests/test_lgn tests/test_glm_q23_metal ds4_test_hooks.o ds4_metal_test_hooks.o quality/score_official quality/score_official.o speed-bench/metal_decode_schedule_bench speed-bench/metal_prefill_variant_bench speed-bench/*.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_mxfp4_metal tests/test_metal_session_batch tests/test_sampling tests/*.o *.o
