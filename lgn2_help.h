#ifndef LGN2_HELP_H
#define LGN2_HELP_H

#include <stdbool.h>
#include <stdio.h>

#ifndef LGN2_BUILD_REVISION
#define LGN2_BUILD_REVISION "unknown"
#endif

#define LGN2_RELEASE_LABEL "development"

typedef enum {
    LGN2_HELP_LGN2,
    LGN2_HELP_SERVER,
    LGN2_HELP_BENCH,
    LGN2_HELP_EVAL,
} lgn2_help_tool;

bool lgn2_help_version_requested(int argc, char *const argv[]);
void lgn2_help_print_version(FILE *fp, lgn2_help_tool tool);
void lgn2_help_print(FILE *fp, lgn2_help_tool tool, const char *topic);

#endif
