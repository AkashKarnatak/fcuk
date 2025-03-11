#include "fcuk.h"

bool isspecial(char c) {
  return c == '/' || c == '-' || c == '_' || c == ' ' || c == '.';
}

void compute_bonus(string_t source, score_t *match_bonus) {
  char prev = '/';
  for (size_t i = 0; i < source.len; ++i) {
    char curr = source.data[i];
    if (islower(curr) && isspecial(prev)) {
      match_bonus[i] = SPECIAL_BONUS[(unsigned char)prev];
    } else if (isupper(curr) && isspecial(prev)) {
      match_bonus[i] = SPECIAL_BONUS[(unsigned char)prev];
    } else if (isupper(curr) && islower(prev)) {
      match_bonus[i] = UPPERCASE_BONUS;
    } else if (isdigit(curr) && isspecial(prev)) {
      match_bonus[i] = SPECIAL_BONUS[(unsigned char)prev];
    } else {
      match_bonus[i] = 0;
    }
    prev = curr;
  }
}

bool has_match(const char *restrict source, const char *restrict pattern) {
  while (*source != '\0' && *pattern != '\0') {
    if (tolower(*source) == tolower(*pattern))
      ++pattern;
    ++source;
  }
  return *pattern == '\0';
}

score_t score(string_t source, string_t pattern) {
  if (source.len >= 1024 || pattern.len >= 1024) {
    // strings too long
    return SCORE_MIN;
  }

  if (source.len == pattern.len) {
    // this function is only called when str contains the
    // pattern
    return SCORE_MAX;
  }

  score_t match_bonus[source.len];
  compute_bonus(source, match_bonus);

  score_t M[pattern.len + 1][source.len + 1];
  score_t D[pattern.len + 1][source.len + 1];

  for (size_t i = 0; i <= pattern.len; ++i) {
    M[i][0] = SCORE_MIN;
  }
  for (size_t j = 0; j <= source.len; ++j) {
    M[0][j] = SCORE_MIN;
  }

  for (size_t i = 1; i <= pattern.len; ++i) {
    score_t gap_penalty =
        i == pattern.len ? GAP_PENALTY_TRAILING : GAP_PENALTY_INNER;

    for (size_t j = 1; j <= source.len; ++j) {
      if (tolower(source.data[j - 1]) == tolower(pattern.data[i - 1])) {
        score_t score;
        if (i == 1) {
          score = (j - 1) * GAP_PENALTY_LEADING + match_bonus[j - 1];
        } else {
          score = M[i - 1][j - 1] + ((D[i - 1][j - 1] != SCORE_MIN)
                                         ? CONSECUTIVE_BONUS
                                         : match_bonus[j - 1]);
        }

        D[i][j] = score;
        M[i][j] = fmax(D[i][j], M[i][j - 1] + gap_penalty);
      } else {
        D[i][j] = SCORE_MIN;
        M[i][j] = M[i][j - 1] + gap_penalty;
      }
    }
  }

  return M[pattern.len][source.len];
}

strings_t match(strings_t *sources, string_t pattern) {
  strings_t matches = {0};
  for (size_t i = 0; i < sources->count; ++i) {
    if (has_match(sources->items[i].data, pattern.data)) {
      da_append(matches, sources->items[i], string_t);
    }
  }
  return matches;
}

results_t score_matches(strings_t *matches, string_t pattern) {
  results_t res = {0};

  for (size_t i = 0; i < matches->count; ++i) {
    scored_entry_t s = {.str = matches->items[i],
                        .score = score(matches->items[i], pattern)};
    da_append(res, s, scored_entry_t);
  }

  return res;
}
