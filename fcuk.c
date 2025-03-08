#include "fcuk.h"

score_t score(const char *restrict str, const char *restrict pattern) {
  size_t n_str, n_ptrn;

  n_str = strlen(str);
  n_ptrn = strlen(pattern);

  assert(n_str <= 1024 && n_ptrn <= 1024);

  score_t match_bonus[n_str];
  compute_bonus(str, n_str, match_bonus);

  score_t M[n_ptrn + 1][n_str + 1];
  score_t D[n_ptrn + 1][n_str + 1];

  for (size_t i = 0; i <= n_ptrn; ++i) {
    M[i][0] = 0;
  }
  for (size_t j = 0; j <= n_str; ++j) {
    M[0][j] = SCORE_MIN;
  }

  for (size_t i = 1; i <= n_ptrn; ++i) {
    score_t gap_penalty =
        i == n_ptrn ? GAP_PENALTY_TRAILING : GAP_PENALTY_INNER;

    for (size_t j = 1; j <= n_str; ++j) {
      if (str[j - 1] == pattern[i - 1]) {
        score_t bonus = (D[i - 1][j - 1] != SCORE_MIN) ? CONSECUTIVE_BONUS
                                                       : match_bonus[j - 1];

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
