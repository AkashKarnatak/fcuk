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

  score_t M[2][source.len];
  score_t D[2][source.len];

  size_t prev = 1,
         curr = 0; // because buffers will be swaped before calculation
  for (size_t i = 0; i < pattern.len; ++i) {
    score_t gap_penalty =
        i == pattern.len - 1 ? GAP_PENALTY_TRAILING : GAP_PENALTY_INNER;
    score_t prev_score = SCORE_MIN;

    // swap buffers before calculating
    size_t tmp = prev;
    prev = curr;
    curr = tmp;

    for (size_t j = 0; j < source.len; ++j) {
      if (tolower(source.data[j]) == tolower(pattern.data[i])) {
        score_t score = SCORE_MIN;
        if (i == 0) {
          score = j * GAP_PENALTY_LEADING + match_bonus[j];
        } else if (j > 0) {
          score = M[prev][j - 1] + ((D[prev][j - 1] != SCORE_MIN)
                                        ? CONSECUTIVE_BONUS
                                        : match_bonus[j]);
        }

        D[curr][j] = score;
        M[curr][j] = prev_score = fmax(D[curr][j], prev_score + gap_penalty);
      } else {
        D[curr][j] = SCORE_MIN;
        M[curr][j] = prev_score = prev_score + gap_penalty;
      }
    }
  }

  return M[curr][source.len - 1];
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
