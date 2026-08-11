#ifndef LGN_DFLASH_EXEC_H
#define LGN_DFLASH_EXEC_H

/* Private DFlash execution boundary.  The context is a borrowed immutable
 * view of engine-owned maps and weights; this module never retains it. */

#include <stdbool.h>
#include <stdint.h>

#include "lgn_dflash_graph.h"

#ifndef LGN2_NO_GPU

enum {
    LGN_DFLASH_TARGET_CONTEXT_LENGTH = 262144u,
};

typedef struct {
    const void *support_map;
    uint64_t    support_map_size;
    const void *f16_map;
    uint64_t    f16_map_size;
    const lgn_dflash_weights *support_weights;
    uint64_t    target_context_length;
} lgn_dflash_exec_context;

bool lgn_dflash_exec_context_valid(
        const lgn_dflash_exec_context *ctx);

/* Record one support-model matrix multiply into the caller's active batch. */
bool lgn_dflash_exec_matmul(
        lgn2_gpu_tensor                 *out,
        const lgn_dflash_exec_context  *ctx,
        const lgn2_tensor               *weight,
        const lgn2_gpu_tensor           *x,
        uint32_t                        n_rows);

/* Record the six-layer support injection into an already-active batch.
 * Ownership of begin/end/submit/discard/wait remains with lgn2_engine.c. */
bool lgn_dflash_exec_encode_record(
        lgn_dflash_graph              *g,
        const lgn_dflash_exec_context *ctx,
        uint32_t                       pos0,
        uint32_t                       n_rows);

#endif /* !LGN2_NO_GPU */

#endif /* LGN_DFLASH_EXEC_H */
