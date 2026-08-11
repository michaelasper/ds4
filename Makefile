CC ?= cc
UNAME_S := $(shell uname -s)
.DEFAULT_GOAL := all

ifneq ($(UNAME_S),Darwin)
$(error Laguna Metal-only build requires Darwin/Apple Metal (got $(UNAME_S)))
endif

NATIVE_CPU_FLAG ?= -mcpu=native
SAMPLING_TEST := tests/test_sampling
METAL_EXACT_TEST := test-laguna-q23-metal

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread
# Metal kernels are loaded from these files at runtime by lgn2_metal.m.  Keep
# the loader's environment-name/default-path pairs explicit for the source
# check below, but do not make the files compile-time prerequisites of the
# large Objective-C objects.
METAL_SOURCE_SPECS := \
	LGN2_METAL_FLASH_ATTN_SOURCE=metal/flash_attn.metal \
	LGN2_METAL_DENSE_SOURCE=metal/dense.metal \
	LGN2_METAL_MOE_SOURCE=metal/moe.metal \
	LGN2_METAL_UNARY_SOURCE=metal/unary.metal \
	LGN2_METAL_DSV4_MISC_SOURCE=metal/dsv4_misc.metal \
	LGN2_METAL_LAGUNA_SOURCE=metal/laguna.metal \
	LGN2_METAL_DFLASH_SOURCE=metal/dflash.metal \
	LGN2_METAL_ARGSORT_SOURCE=metal/argsort.metal \
	LGN2_METAL_CPY_SOURCE=metal/cpy.metal \
	LGN2_METAL_GET_ROWS_SOURCE=metal/get_rows.metal \
	LGN2_METAL_GLU_SOURCE=metal/glu.metal \
	LGN2_METAL_NORM_SOURCE=metal/norm.metal \
	LGN2_METAL_BIN_SOURCE=metal/bin.metal
PREFIX ?= /usr/local
DESTDIR ?=
INSTALL ?= install
LGN2_BINS := lgn2 lgn2-server lgn2-bench lgn2-eval
METAL_INSTALL_DIR := $(PREFIX)/share/lgn2/metal
LGN2_TEST_MODEL ?= lgn2.gguf
LGN2_TEST_DFLASH ?=

METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = lgn2_engine.o lgn.o lgn_model.o lgn_dflash.o lgn_graph.o lgn_dflash_graph.o lgn_dflash_exec.o lgn2_metal.o

LGN2_TEST_METAL_OBJ := lgn2_metal_test_hooks.o
LGN2_TEST_ENGINE_OBJ := lgn2_engine_test_hooks.o
TEST_CORE_OBJS := $(filter-out lgn2_engine.o lgn2_metal.o,$(CORE_OBJS)) $(LGN2_TEST_ENGINE_OBJ) $(LGN2_TEST_METAL_OBJ)

METAL_SOURCE_ORDER_ONLY := | check-metal-sources

.PHONY: all help clean install uninstall test test-extended test-engine-lifecycle test-metal-laguna test-metal-laguna-integration test-laguna-cli-options test-lgn check-metal-sources test-metal-session-batch test-laguna-q23-metal test-installed-resources dflash-verify-depth

# Keep this check cheap and always current: the executable contains only the
# host-side loader, while these source files are read and compiled at runtime.
# A fixed pair list means a missing default cannot disappear from a wildcard;
# an explicitly nonempty override is authoritative and the default is not
# consulted, matching the fail-closed runtime override contract.
check-metal-sources:
	@python3 metal/generate_mxfp4_half_lut.py --check
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

all: check-metal-sources lgn2 lgn2-server lgn2-bench lgn2-eval

install: all
	@set -eu; \
	$(INSTALL) -d "$(DESTDIR)$(PREFIX)/bin" "$(DESTDIR)$(METAL_INSTALL_DIR)"; \
	for bin in $(LGN2_BINS); do \
		$(INSTALL) -m 755 "$$bin" "$(DESTDIR)$(PREFIX)/bin/$$bin"; \
	done; \
	for spec in $(METAL_SOURCE_SPECS); do \
		src=$${spec#*=}; \
		$(INSTALL) -m 644 "$$src" "$(DESTDIR)$(METAL_INSTALL_DIR)/$${src##*/}"; \
	done

uninstall:
	@set -eu; \
	for bin in $(LGN2_BINS); do \
		rm -f "$(DESTDIR)$(PREFIX)/bin/$$bin"; \
	done; \
	for spec in $(METAL_SOURCE_SPECS); do \
		src=$${spec#*=}; \
		rm -f "$(DESTDIR)$(METAL_INSTALL_DIR)/$${src##*/}"; \
	done; \
	rmdir "$(DESTDIR)$(METAL_INSTALL_DIR)" 2>/dev/null || true; \
	rmdir "$(DESTDIR)$(PREFIX)/share/lgn2" 2>/dev/null || true

help:
	@echo "LGN2 build targets:"
	@echo "  make              Build Metal ./lgn2, ./lgn2-server, ./lgn2-bench, and ./lgn2-eval"
	@echo "  make test         Build/run the model-independent Apple Metal/Laguna suite"
	@echo "  make test-metal-laguna  Run the strict model-independent Apple Metal/Laguna suite"
	@echo "  make test-extended  Run the extended model-independent developer suite"
	@echo "  make test-metal-laguna-integration LGN2_TEST_MODEL=FILE  Run model-backed Laguna smoke"
	@echo "  make test-installed-resources  Stage-install sources and test relocated lookup"
	@echo "  make install PREFIX=DIR DESTDIR=DIR  Install binaries and runtime Metal sources"
	@echo "  make uninstall PREFIX=DIR DESTDIR=DIR  Remove installed binaries and runtime sources"
	@echo "  make dflash-verify-depth  Run DFlash speculative verification smoke if support GGUF is present"
	@echo "  make clean        Remove build outputs"

lgn2: lgn2_cli.o lgn2_help.o linenoise.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ lgn2_cli.o lgn2_help.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

lgn2-server: lgn2_server.o lgn2_help.o lgn2_kvstore.o rax.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ lgn2_server.o lgn2_help.o lgn2_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

lgn2-bench: lgn2_bench.o lgn2_help.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ lgn2_bench.o lgn2_help.o $(CORE_OBJS) $(METAL_LDLIBS)

lgn2-eval: lgn2_eval.o lgn2_help.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ lgn2_eval.o lgn2_help.o $(CORE_OBJS) $(METAL_LDLIBS)

quality/score_official: quality/score_official.c lgn2.h $(CORE_OBJS) rax.o | check-metal-sources
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ quality/score_official.c $(CORE_OBJS) rax.o $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c lgn2.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS) | check-metal-sources
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	LGN2_TEST_MODEL="$(LGN2_TEST_MODEL)" ./tests/test_metal_session_batch

tests/test_laguna_q23_metal.o: tests/test_laguna_q23_metal.c lgn2_gpu.h
	$(CC) $(CFLAGS) -DLGN2_TEST_HOOKS -I. -c -o $@ $<

tests/test_laguna_q23_metal: tests/test_laguna_q23_metal.o lgn2_metal_test_hooks.o | check-metal-sources
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-laguna-q23-metal: tests/test_laguna_q23_metal
	./tests/test_laguna_q23_metal

tests/test_sampling.o: tests/test_sampling.c lgn2.h
	$(CC) $(CFLAGS) -DLGN2_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o lgn2_cpu_test_hooks.o lgn.o lgn_model.o lgn_dflash.o lgn2_kvstore.o rax.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_lgn.o: tests/test_lgn.c lgn2.h lgn.h lgn_model.h lgn_dflash.h
	$(CC) $(CFLAGS) -DLGN2_TEST_HOOKS -I. -c -o $@ $<

tests/test_lgn: tests/test_lgn.o lgn2_cpu_test_hooks.o lgn.o lgn_model.o lgn_dflash.o lgn2_kvstore.o rax.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test-lgn: tests/test_lgn
	./tests/test_lgn

lgn2_engine.o: lgn2_engine.c lgn2.h lgn2_gpu.h lgn.h lgn_model.h lgn_dflash.h lgn_graph.h lgn_dflash_graph.h lgn_dflash_exec.h
	$(CC) $(CFLAGS) -c -o $@ lgn2_engine.c

lgn.o: lgn.c lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn.c

lgn_model.o: lgn_model.c lgn_model.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_model.c

lgn_dflash.o: lgn_dflash.c lgn_dflash.h lgn_model.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_dflash.c

lgn_graph.o: lgn_graph.c lgn_graph.h lgn2_gpu.h lgn_model.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_graph.c

lgn_dflash_graph.o: lgn_dflash_graph.c lgn_dflash_graph.h lgn_dflash.h lgn_model.h lgn2_gpu.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_dflash_graph.c

lgn_dflash_exec.o: lgn_dflash_exec.c lgn_dflash_exec.h lgn_dflash_graph.h lgn_dflash.h lgn_model.h lgn2_gpu.h lgn.h
	$(CC) $(CFLAGS) -c -o $@ lgn_dflash_exec.c

lgn2_cli.o: lgn2_cli.c lgn2.h lgn2_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ lgn2_cli.c

lgn2_help.o: lgn2_help.c lgn2_help.h
	$(CC) $(CFLAGS) -c -o $@ lgn2_help.c

lgn2_server.o: lgn2_server.c lgn2.h lgn2_help.h lgn2_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ lgn2_server.c

lgn2_bench.o: lgn2_bench.c lgn2.h lgn2_help.h
	$(CC) $(CFLAGS) -c -o $@ lgn2_bench.c

lgn2_eval.o: lgn2_eval.c lgn2.h lgn2_help.h
	$(CC) $(CFLAGS) -c -o $@ lgn2_eval.c

lgn2_kvstore.o: lgn2_kvstore.c lgn2_kvstore.h lgn2.h
	$(CC) $(CFLAGS) -c -o $@ lgn2_kvstore.c

lgn2_test.o: tests/lgn2_test.c lgn2_server.c lgn2.h lgn2_gpu.h lgn2_help.h lgn2_kvstore.h rax.h lgn.h lgn_graph.h lgn_dflash_graph.h lgn_dflash_exec.h
	$(CC) $(CFLAGS) -Wno-unused-function -DLGN2_TEST_HOOKS -c -o $@ tests/lgn2_test.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

lgn2_metal.o: lgn2_metal.m lgn2_gpu.h | check-metal-sources
	$(CC) $(OBJCFLAGS) -c -o $@ lgn2_metal.m

lgn2_metal_test_hooks.o: lgn2_metal.m lgn2_gpu.h | check-metal-sources
	$(CC) $(OBJCFLAGS) -DLGN2_TEST_HOOKS -c -o $@ lgn2_metal.m

lgn2_engine_test_hooks.o: lgn2_engine.c lgn2.h lgn2_gpu.h lgn.h lgn_model.h lgn_dflash.h lgn_graph.h lgn_dflash_graph.h lgn_dflash_exec.h
	$(CC) $(CFLAGS) -Wno-unused-function -DLGN2_TEST_HOOKS -c -o $@ lgn2_engine.c

lgn2_cpu_test_hooks.o: lgn2_engine.c lgn2.h lgn2_gpu.h lgn.h lgn_model.h lgn_dflash.h lgn_graph.h lgn_dflash_graph.h lgn_dflash_exec.h
	$(CC) $(CFLAGS) -Wno-unused-function -DLGN2_NO_GPU -DLGN2_TEST_HOOKS -c -o $@ lgn2_engine.c

lgn2_test: lgn2_test.o lgn2_help.o lgn2_kvstore.o rax.o $(TEST_CORE_OBJS) $(METAL_SOURCE_ORDER_ONLY)
	$(CC) $(CFLAGS) -o $@ lgn2_test.o lgn2_help.o lgn2_kvstore.o rax.o $(TEST_CORE_OBJS) $(METAL_LDLIBS)

test-engine-lifecycle: lgn2_test
	./lgn2_test --engine-lifecycle

test-extended: test-lgn lgn2-eval q4k-dot-test mxfp4-dot-test \
	$(SAMPLING_TEST) $(METAL_EXACT_TEST) lgn2 lgn2-server lgn2-bench test-engine-lifecycle
	./lgn2-eval --self-test-extractors
	./tests/test_sampling

test-laguna-cli-options: lgn2 lgn2-server lgn2-bench lgn2-eval tests/test_laguna_cli_options.sh
	./tests/test_laguna_cli_options.sh

test-metal-laguna: check-metal-sources test-lgn test-laguna-q23-metal test-laguna-cli-options test-engine-lifecycle $(SAMPLING_TEST) lgn2 lgn2-server lgn2-bench lgn2-eval
	@set -eu; \
	./lgn2_test --laguna-moe-abi --laguna-architecture --laguna-session-surface --laguna-selector-parser --laguna-session-routes --laguna-graph-lifecycle --laguna-dflash-graph-lifecycle --laguna-dflash-exec --laguna-dflash-command-ownership --server; \
	LGN2_METAL_MOE_SOURCE=metal/moe.metal LGN2_TEST_MOE_ABI_MODE=current ./lgn2_test --laguna-moe-abi; \
	./lgn2_test --dflash-payload-lifecycle; \
	./tests/test_sampling; \
	LGN2_TEST_LAGUNA_STAGED_SWA_ALLOW_FALLBACK= \
	./lgn2_test --laguna-metal-core

test-installed-resources: all lgn2_test tests/test_lgn2_resource_install.sh
	@set -eu; \
	stage=$$(mktemp -d "$${TMPDIR:-/tmp}/lgn2-install-test.XXXXXX"); \
	trap 'rm -rf "$$stage"' EXIT HUP INT TERM; \
	$(MAKE) --no-print-directory install DESTDIR="$$stage" PREFIX=/usr/local; \
	cp -p lgn2_test "$$stage/usr/local/bin/lgn2_test"; \
	./tests/test_lgn2_resource_install.sh "$$stage/usr/local" "$$stage/usr/local/bin/lgn2_test"; \
	rm -f "$$stage/usr/local/bin/lgn2_test"; \
	$(MAKE) --no-print-directory uninstall DESTDIR="$$stage" PREFIX=/usr/local; \
	for bin in $(LGN2_BINS); do test ! -e "$$stage/usr/local/bin/$$bin"; done; \
	test ! -e "$$stage/usr/local/share/lgn2/metal"

test: test-metal-laguna

test-metal-laguna-integration: check-metal-sources lgn2 lgn2-server lgn2-bench lgn2-eval tests/test_metal_session_batch
	@test -n "$(strip $(LGN2_TEST_MODEL))" || { \
		echo "error: set LGN2_TEST_MODEL=/path/to/laguna-s2.1.gguf for model-backed integration" >&2; \
		exit 2; \
	}
	@test -f "$(LGN2_TEST_MODEL)" || { \
		echo "error: Laguna integration model not found: $(LGN2_TEST_MODEL)" >&2; \
		exit 2; \
	}
	LGN2_TEST_MODEL="$(LGN2_TEST_MODEL)" ./lgn2 --metal --model "$(LGN2_TEST_MODEL)" --inspect
	LGN2_TEST_MODEL="$(LGN2_TEST_MODEL)" \
	./tests/test_metal_session_batch

dflash-verify-depth: lgn2_test
	@if [ ! -f "$(LGN2_TEST_MODEL)" ]; then \
		echo "dflash-verify-depth: skipped, missing model $(LGN2_TEST_MODEL)"; \
	elif [ -z "$(strip $(LGN2_TEST_DFLASH))" ]; then \
		echo "dflash-verify-depth: skipped, set LGN2_TEST_DFLASH=FILE to a DFlash support GGUF"; \
	elif [ ! -f "$(LGN2_TEST_DFLASH)" ]; then \
		echo "dflash-verify-depth: skipped, missing DFlash support $(LGN2_TEST_DFLASH)"; \
	else \
		LGN2_TEST_MODEL="$(LGN2_TEST_MODEL)" LGN2_TEST_DFLASH="$(LGN2_TEST_DFLASH)" ./lgn2_test --dflash-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

clean:
	rm -f lgn2 lgn2-server lgn2-bench lgn2-eval lgn2_test tests/test_lgn tests/test_laguna_q23_metal lgn2_engine_test_hooks.o lgn2_metal_test_hooks.o quality/score_official quality/score_official.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_metal_session_batch tests/test_sampling tests/*.o *.o
