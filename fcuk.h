#ifndef COMMON_H
#define COMMON_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

score_t score(const char *__restrict__ str, const char *__restrict__ pattern);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
