#include "fcuk.h"

const double SPECIAL_BONUS[256] = {['/'] = SLASH_BONUS,
                                   ['-'] = DASH_BONUS,
                                   ['_'] = UNDERSCORE_BONUS,
                                   [' '] = SPACE_BONUS,
                                   ['.'] = DOT_BONUS};

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

bool match(const char *restrict str, const char *restrict pattern) {
  assert(str && pattern);
  while (*str != '\0' && *pattern != '\0') {
    if (tolower(*str) == tolower(*pattern))
      ++pattern;
    ++str;
  }
  return *pattern == '\0';
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
