#ifndef COMMON_H
#define COMMON_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void compute_bonus(const char *__restrict__ str, size_t n,
                   score_t *__restrict__ match_bonus);

score_t score(const char *__restrict__ str, const char *__restrict__ pattern);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
