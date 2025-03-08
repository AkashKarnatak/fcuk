#include "common.h"

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
      match_bonus[i] = SPECIAL_BONUS[(unsigned char)prev];
    } else if (isupper(curr) && isspecial(prev)) {
      match_bonus[i] = SPECIAL_BONUS[(unsigned char)prev];
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
