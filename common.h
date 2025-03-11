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

typedef float score_t;

#define SCORE_MIN -INFINITY
#define SCORE_MAX INFINITY
#define CONSECUTIVE_BONUS 1.0
#define GAP_PENALTY_LEADING -0.005
#define GAP_PENALTY_INNER -0.01
#define GAP_PENALTY_TRAILING -0.005
#define UPPERCASE_BONUS 0.7
#define SLASH_BONUS 0.9
#define DASH_BONUS 0.8
#define UNDERSCORE_BONUS 0.8
#define SPACE_BONUS 0.8
#define DOT_BONUS 0.6

#define da_append(xs, x, T)                                                    \
  do {                                                                         \
    if ((xs).count >= (xs).capacity) {                                         \
      if ((xs).capacity == 0)                                                  \
        (xs).capacity = 256;                                                   \
      else                                                                     \
        (xs).capacity *= 2;                                                    \
      (xs).items =                                                             \
          (T *)realloc((xs).items, (xs).capacity * sizeof(*(xs).items));       \
    }                                                                          \
    (xs).items[(xs).count++] = (x);                                            \
  } while (0)

typedef struct {
  char *data;
  size_t len;
} string_t;

typedef struct {
  string_t *items;
  size_t count;
  size_t capacity;
} strings_t;

typedef struct {
  string_t str;
  score_t score;
} scored_entry_t;

typedef struct {
  scored_entry_t *items;
  size_t count;
  size_t capacity;
} results_t;

extern const score_t SPECIAL_BONUS[256];

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif
