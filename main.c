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

const double SPECIAL_BONUS[256] = {['/'] = SLASH_BONUS,
                                   ['-'] = DASH_BONUS,
                                   ['_'] = UNDERSCORE_BONUS,
                                   [' '] = SPACE_BONUS,
                                   ['.'] = DOT_BONUS};

#define da_append(xs, x)                                                       \
  do {                                                                         \
    if (xs.count >= xs.capacity) {                                             \
      if (xs.capacity == 0)                                                    \
        xs.capacity = 256;                                                     \
      else                                                                     \
        xs.capacity *= 2;                                                      \
      xs.items = realloc(xs.items, xs.capacity * sizeof(*xs.items));           \
    }                                                                          \
    xs.items[xs.count++] = x;                                                  \
  } while (0)

bool match(const char *restrict str, const char *restrict pattern) {
  assert(str && pattern);
  while (*str != '\0' && *pattern != '\0') {
    if (tolower(*str) == tolower(*pattern))
      ++pattern;
    ++str;
  }
  return *pattern == '\0';
}

bool isspecial(char c) {
  return c == '/' || c == '-' || c == '_' || c == ' ' || c == '.';
}

void compute_bonus(const char *restrict str, size_t n, score_t *match_bonus) {
  char prev = '/';
  for (size_t i = 0; i < n; ++i) {
    char curr = str[i];
    if (islower(curr) && isspecial(prev)) {
      match_bonus[i] = SPECIAL_BONUS[prev];
    } else if (isupper(curr) && isspecial(prev)) {
      match_bonus[i] = SPECIAL_BONUS[prev];
    } else if (isupper(curr) && islower(prev)) {
      match_bonus[i] = UPPERCASE_BONUS;
    } else {
      match_bonus[i] = 0;
    }
  }
}

score_t score(const char *restrict str, const char *restrict pattern) {
  size_t n_str, n_ptrn;

  n_str = strlen(str);
  n_ptrn = strlen(pattern);

  assert(n_str <= 1024 && n_ptrn <= 1024);

  score_t M[n_ptrn + 1][n_str + 1];
  bool D[n_ptrn + 1][n_str + 1];

  score_t match_bonus[n_str];
  compute_bonus(str, n_str, match_bonus);

  for (size_t i = 1; i <= n_ptrn; ++i) {
    score_t gap_penalty =
        i == n_ptrn ? GAP_PENALTY_TRAILING : GAP_PENALTY_INNER;

    for (size_t j = 1; j <= n_str; ++j) {
      if (str[j - 1] == pattern[i - 1]) {
        score_t bonus =
            (D[i - 1][j - 1] != SCORE_MIN) ? CONSECUTIVE_BONUS : match_bonus[j];

        D[i][j] = M[i - 1][j - 1] + bonus;
        M[i][j] = fmax(D[i][j], M[i][j - 1] + gap_penalty);
      } else {
        D[i][j] = SCORE_MIN;
        M[i][j] = M[i][j - 1] + gap_penalty;
      }
    }
  }

  return M[n_ptrn][n_str];
}

typedef struct {
  char **items;
  size_t count;
  size_t capacity;
} Strings;

int main(int argc, char *argv[]) {
  char *buf, *pattern;
  size_t buf_size;
  int32_t n;
  Strings strings = {0};

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <PATTERN>\n", argv[0]);
    exit(1);
  }

  pattern = argv[1];

  buf = NULL;
  while ((n = getline(&buf, &buf_size, stdin)) != -1) {
    buf[n - 1] = '\0';
    da_append(strings, buf);
    buf = NULL;
  }
  free(buf);

  for (size_t i = 0; i < strings.count; ++i) {
    if (match(strings.items[i], pattern)) {
      printf("%s: %lf\n", strings.items[i], score(strings.items[i], pattern));
    }
  }

  // clean up
  for (size_t i = 0; i < strings.count; ++i) {
    free(strings.items[i]);
  }
  free(strings.items);
}
