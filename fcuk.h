#ifndef COMMON_H
#define COMMON_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

strings_t match(strings_t *sources, string_t pattern);

results_t score_matches(strings_t *matches, string_t pattern);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
