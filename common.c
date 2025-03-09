#include "common.h"

const score_t SPECIAL_BONUS[256] = {['/'] = SLASH_BONUS,
                                   ['-'] = DASH_BONUS,
                                   ['_'] = UNDERSCORE_BONUS,
                                   [' '] = SPACE_BONUS,
                                   ['.'] = DOT_BONUS};

bool match(const char *restrict str, const char *restrict pattern) {
  assert(str && pattern);
  while (*str != '\0' && *pattern != '\0') {
    if (tolower(*str) == tolower(*pattern))
      ++pattern;
    ++str;
  }
  return *pattern == '\0';
}
