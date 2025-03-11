#include "fcuk.h"

__device__ inline bool islower(char c) { return c >= 'a' && c <= 'z'; }

__device__ inline bool isupper(char c) { return c >= 'A' && c <= 'Z'; }

__device__ inline bool isdigit(char c) { return c >= '0' && c <= '9'; }

__device__ inline bool isspecial(char c) {
  return c == '/' || c == '-' || c == '_' || c == ' ' || c == '.';
}

__device__ inline char tolower(char c) {
  return isupper(c) ? c - 'A' + 'a' : c;
}

bool init = false;
__constant__ score_t SPECIAL_BONUS_C[256];

__global__ void match_kernel(const char *__restrict__ str,
                             const char *__restrict__ pattern, size_t n_str,
                             size_t n_ptrn) {
  size_t j = threadIdx.x;
  __shared__ size_t idx[MAX_STR_LEN]; // TODO: wasting resource, change to n_str
  __shared__ char c;
  int32_t prev = -1;

  for (size_t i = 0; i < n_ptrn; ++i) {
    if (threadIdx.x == 0) {
      c = pattern[i];
    }
    __syncthreads();

    idx[j] = c == str[j] && j > prev ? j : INT32_MAX;

    // calculate min using parallel reduction
    for (size_t s = blockDim.x / 2; s >= 1; s /= 2) {
      if (j < s) {
        idx[j] = min(idx[j], idx[j + s]);
      }
      __syncthreads();
    }

    prev = idx[0];

    if (prev == INT32_MAX) {
      // TODO return false
      return;
    }
  }
}

__global__ void fused_score_kernel(const char *__restrict__ str,
                                   const char *__restrict__ pattern, score_t *M,
                                   score_t *D, size_t n_str, size_t n_ptrn) {
  size_t j = threadIdx.x;
  __shared__ score_t
      match_bonus_s[MAX_STR_LEN]; // TODO: wasting resource, change to n_str

  if (j >= 1 && j <= n_str) {
    char curr = str[j - 1], prev = j > 1 ? str[j - 2] : '/';

    if (islower(curr) && isspecial(prev)) {
      match_bonus_s[j - 1] = SPECIAL_BONUS_C[(unsigned char)prev];
    } else if (isupper(curr) && isspecial(prev)) {
      match_bonus_s[j - 1] = SPECIAL_BONUS_C[(unsigned char)prev];
    } else if (isupper(curr) && islower(prev)) {
      match_bonus_s[j - 1] = UPPERCASE_BONUS;
    } else if (isdigit(curr) && isspecial(prev)) {
      match_bonus_s[j - 1] = SPECIAL_BONUS_C[(unsigned char)prev];
    } else {
      match_bonus_s[j - 1] = 0;
    }
  }

  __syncthreads();

  for (size_t wave = 0; wave <= (n_str + n_ptrn); ++wave) {
    int32_t i = wave - j;
    if (i >= 0 && i <= n_ptrn) {
      if (i == 0) {
        M[j] = SCORE_MIN;
        continue;
      }
      if (j == 0) {
        M[i * (n_str + 1)] = SCORE_MIN;
        continue;
      }

      score_t gap_penalty =
          i == n_ptrn ? GAP_PENALTY_TRAILING : GAP_PENALTY_INNER;

      if (tolower(str[j - 1]) == tolower(pattern[i - 1])) {
        score_t score;
        if (i == 1) {
          score = (j - 1) * GAP_PENALTY_LEADING + match_bonus_s[j - 1];
        } else {
          score = M[(i - 1) * (n_str + 1) + j - 1] +
                  ((D[(i - 1) * (n_str + 1) + j - 1] != SCORE_MIN)
                       ? CONSECUTIVE_BONUS
                       : match_bonus_s[j - 1]);
        }

        D[i * (n_str + 1) + j] = score;
        M[i * (n_str + 1) + j] = max(D[i * (n_str + 1) + j],
                                     M[i * (n_str + 1) + j - 1] + gap_penalty);
      } else {
        D[i * (n_str + 1) + j] = SCORE_MIN;
        M[i * (n_str + 1) + j] = M[i * (n_str + 1) + j - 1] + gap_penalty;
      }
    }
    __syncthreads();
  }
}

bool has_match(const char *__restrict__ source,
               const char *__restrict__ pattern) {
  while (*source != '\0' && *pattern != '\0') {
    if (tolower((int)*source) == tolower((int)*pattern))
      ++pattern;
    ++source;
  }
  return *pattern == '\0';
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

score_t score(string_t source, string_t pattern) {
  char *str_d, *pattern_d;
  score_t res, *M_d, *D_d;

  if (source.len >= 1024 || pattern.len >= 1024) {
    // strings too long
    return SCORE_MIN;
  }

  if (source.len == pattern.len) {
    // this function is only called when str contains the
    // pattern
    return SCORE_MAX;
  }

  // TODO: figure out a better way
  if (!init) {
    cudaMemcpyToSymbol(SPECIAL_BONUS_C, SPECIAL_BONUS, sizeof(SPECIAL_BONUS));
    init = true;
  }

  cudaMalloc(&str_d, source.len * sizeof(char));
  cudaMalloc(&pattern_d, pattern.len * sizeof(char));
  cudaMalloc(&M_d, (pattern.len + 1) * (source.len + 1) * sizeof(score_t));
  cudaMalloc(&D_d, (pattern.len + 1) * (source.len + 1) * sizeof(score_t));

  cudaMemcpy(str_d, source.data, source.len * sizeof(char),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pattern_d, pattern.data, pattern.len * sizeof(char),
             cudaMemcpyHostToDevice);

  dim3 numThreads(source.len + 1);
  dim3 numBlocks(1);
  fused_score_kernel<<<numBlocks, numThreads>>>(str_d, pattern_d, M_d, D_d,
                                                source.len, pattern.len);

  cudaMemcpy(&res, &M_d[pattern.len * (source.len + 1) + source.len],
             sizeof(score_t), cudaMemcpyDeviceToHost);

  cudaFree(str_d);
  cudaFree(pattern_d);
  cudaFree(M_d);
  cudaFree(D_d);

  return res;
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
