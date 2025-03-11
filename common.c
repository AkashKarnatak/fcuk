#include "common.h"

const score_t SPECIAL_BONUS[256] = {['/'] = SLASH_BONUS,
                                   ['-'] = DASH_BONUS,
                                   ['_'] = UNDERSCORE_BONUS,
                                   [' '] = SPACE_BONUS,
                                   ['.'] = DOT_BONUS};
