#ifndef LGN2_HELP_H
#define LGN2_HELP_H

#include <stdio.h>

typedef enum {
    LGN2_HELP_LGN2,
    LGN2_HELP_SERVER,
    LGN2_HELP_BENCH,
    LGN2_HELP_EVAL,
} lgn2_help_tool;

void lgn2_help_print(FILE *fp, lgn2_help_tool tool, const char *topic);

#endif
