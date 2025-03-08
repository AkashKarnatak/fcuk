#ifndef FCUK_H
#define FCUK_H

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STR_LEN 1024

typedef double score_t;

#define SCORE_MIN -INFINITY
#define SCORE_MAX INFINITY
#define CONSECUTIVE_BONUS 1.0
#define GAP_PENALTY_INNER -0.005
#define GAP_PENALTY_TRAILING -0.01
#define UPPERCASE_BONUS 0.7
#define SLASH_BONUS 0.9
#define DASH_BONUS 0.8
#define UNDERSCORE_BONUS 0.8
#define SPACE_BONUS 0.8
#define DOT_BONUS 0.6

#ifdef __cplusplus
extern "C" {
#endif

bool isspecial(char c);

void compute_bonus(const char *__restrict__ str, size_t n,
                   score_t *match_bonus);

bool match(const char *__restrict__ str, const char *__restrict__ pattern);

score_t score(const char *__restrict__ str, const char *__restrict__ pattern);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
