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

__global__ void match_kernel(char *__restrict__ buf,
                             size_t *__restrict__ indices, size_t buf_size,
                             size_t n_sources, const char *__restrict__ pattern,
                             size_t n_pattern, bool *has_match) {
  size_t idx = indices[blockIdx.x];
  string_t source = {
      .data = buf + idx,
      .len =
          (blockIdx.x == n_sources - 1 ? buf_size : indices[blockIdx.x + 1]) -
          idx};
  int32_t j = threadIdx.x;

  if (j >= source.len)
    return;

  // // sequential
  // if (j == 0) {
  //   for (size_t i = 0; i < source.len; ++i) {
  //     if (tolower(source.data[i]) == *pattern)
  //       ++pattern;
  //   }
  //   has_match[blockIdx.x] = *pattern == '\0';
  // }

  if (j == 0) {
    has_match[blockIdx.x] = true;
  }

  __shared__ size_t pos[MAX_STR_LEN]; // TODO: wasting resource, change to n_str
  int32_t prev = -1;

  for (size_t i = 0; i < n_pattern; ++i) {
    pos[j] = tolower(source.data[j]) == tolower(pattern[i]) && j > prev
                 ? j
                 : INT32_MAX;

    __syncthreads();

    for (size_t n = source.len, s = (n + 1) / 2; s > 0;
         n = s, s = (s + 1) / 2) {
      if (j < s && j + s < n) {
        pos[j] = min(pos[j], pos[j + s]);
      }
      __syncthreads();
      if (s == 1)
        break;
    }

    prev = pos[0];

    if (prev == INT32_MAX) {
      if (j == 0) {
        has_match[blockIdx.x] = false;
      }
      return;
    }

    __syncthreads();
  }
}

__global__ void fused_score_kernel(char *__restrict__ buf,
                                   size_t *__restrict__ indices,
                                   size_t buf_size, size_t n_sources,
                                   const char *__restrict__ pattern,
                                   size_t n_pattern, score_t *res_scores) {
  size_t idx = indices[blockIdx.x];
  string_t source = {
      .data = buf + idx,
      .len =
          (blockIdx.x == n_sources - 1 ? buf_size : indices[blockIdx.x + 1]) -
          idx};
  size_t j = threadIdx.x;

  if (j >= source.len)
    return;

  if (source.len > 1024 || n_pattern > 1024) {
    // strings too long
    if (j == 0) {
      res_scores[blockIdx.x] = SCORE_MIN;
    }
    return;
  }

  if (source.len == n_pattern) {
    // this function is only called when str contains the
    // pattern
    if (j == 0) {
      res_scores[blockIdx.x] = SCORE_MAX;
    }
    return;
  }

  __shared__ score_t
      match_bonus_s[MAX_STR_LEN]; // TODO: wasting resource, change to n_str
  __shared__ score_t M_s[3][MAX_STR_LEN], D_s[3][MAX_STR_LEN];

  M_s[0][j] = SCORE_MIN;
  M_s[1][j] = SCORE_MIN;
  M_s[2][j] = SCORE_MIN;

  D_s[0][j] = SCORE_MIN;
  D_s[1][j] = SCORE_MIN;
  D_s[2][j] = SCORE_MIN;

  if (j < source.len) {
    char curr = source.data[j], prev = j > 0 ? source.data[j - 1] : '/';

    if (islower(curr) && isspecial(prev)) {
      match_bonus_s[j] = SPECIAL_BONUS_C[(unsigned char)prev];
    } else if (isupper(curr) && isspecial(prev)) {
      match_bonus_s[j] = SPECIAL_BONUS_C[(unsigned char)prev];
    } else if (isupper(curr) && islower(prev)) {
      match_bonus_s[j] = UPPERCASE_BONUS;
    } else if (isdigit(curr) && isspecial(prev)) {
      match_bonus_s[j] = SPECIAL_BONUS_C[(unsigned char)prev];
    } else {
      match_bonus_s[j] = 0;
    }
  }

  __syncthreads();

  size_t prev2 = 0, prev = 1, curr = 2;
  for (size_t wave = 0; wave <= (source.len - 1 + n_pattern - 1); ++wave) {
    int32_t i = wave - j;

    // swap offset before calculating
    curr = (curr + 1) % 3;
    prev = (prev + 1) % 3;
    prev2 = (prev2 + 1) % 3;

    if (i >= 0 && i < n_pattern) {
      score_t gap_penalty =
          i == n_pattern - 1 ? GAP_PENALTY_TRAILING : GAP_PENALTY_INNER;

      if (tolower(source.data[j]) == tolower(pattern[i])) {
        score_t score = SCORE_MIN;
        if (i == 0) {
          score = j * GAP_PENALTY_LEADING + match_bonus_s[j];
        } else if (j > 0) {
          score = M_s[prev2][j - 1] + ((D_s[prev2][j - 1] != SCORE_MIN)
                                           ? CONSECUTIVE_BONUS
                                           : match_bonus_s[j]);
        }

        D_s[curr][j] = score;
        M_s[curr][j] = fmax(
            D_s[curr][j], (j > 0 ? M_s[prev][j - 1] : SCORE_MIN) + gap_penalty);
      } else {
        D_s[curr][j] = SCORE_MIN;
        M_s[curr][j] = (j > 0 ? M_s[prev][j - 1] : SCORE_MIN) + gap_penalty;
      }
    }
    __syncthreads();
  }

  if (j == 0) {
    res_scores[blockIdx.x] = M_s[curr][source.len - 1];
  }
}

strings_t match(strings_t *sources, string_t pattern) {
  // TODO: figure out a better way
  if (!init) {
    cudaMemcpyToSymbol(SPECIAL_BONUS_C, SPECIAL_BONUS, sizeof(SPECIAL_BONUS));
    init = true;
  }

  strings_t matches = {0};

  size_t buf_size = 0;
  for (size_t i = 0; i < sources->count; ++i) {
    buf_size += sources->items[i].len;
  }

  char *buf_h, *buf_h_itr; // pinned memory
  size_t *indices_h;       // pinned memory
  bool *has_match_h;       // pinned memory
  char *buf_d, *pattern_d;
  size_t *indices_d;
  bool *has_match_d;
  size_t prev_sum, n_threads;

  cudaMallocHost(&buf_h, buf_size * sizeof(char));
  cudaMallocHost(&indices_h, sources->count * sizeof(size_t));
  cudaMallocHost(&has_match_h, sources->count * sizeof(bool));

  cudaMalloc(&buf_d, buf_size * sizeof(char));
  cudaMalloc(&indices_d, sources->count * sizeof(size_t));
  cudaMalloc(&pattern_d, pattern.len * sizeof(char));
  cudaMalloc(&has_match_d, sources->count * sizeof(bool));

  buf_h_itr = buf_h;
  prev_sum = 0, n_threads = 0;
  for (size_t i = 0; i < sources->count; ++i) {
    memcpy(buf_h_itr, sources->items[i].data,
           sources->items[i].len * sizeof(char));
    buf_h_itr += sources->items[i].len;

    indices_h[i] = prev_sum;
    prev_sum += sources->items[i].len;

    n_threads =
        sources->items[i].len > n_threads ? sources->items[i].len : n_threads;
  }
  n_threads = 1024 < n_threads ? 1024 : n_threads;

  cudaMemcpy(buf_d, buf_h, buf_size * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(indices_d, indices_h, sources->count * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pattern_d, pattern.data, pattern.len * sizeof(char),
             cudaMemcpyHostToDevice);

  dim3 numThreads(n_threads);
  dim3 numBlocks(sources->count);
  match_kernel<<<numBlocks, numThreads>>>(buf_d, indices_d, buf_size,
                                          sources->count, pattern_d,
                                          pattern.len, has_match_d);

  cudaMemcpy(has_match_h, has_match_d, sources->count * sizeof(bool),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < sources->count; ++i) {
    if (has_match_h[i])
      da_append(matches, sources->items[i], string_t);
  }

  cudaFree(buf_h);
  cudaFree(indices_h);
  cudaFree(has_match_h);

  cudaFree(buf_d);
  cudaFree(indices_d);
  cudaFree(pattern_d);
  cudaFree(has_match_d);

  return matches;
}

results_t score_matches(strings_t *__restrict__ matches, string_t pattern) {
  results_t res = {0};

  size_t buf_size = 0;
  for (size_t i = 0; i < matches->count; ++i) {
    buf_size += matches->items[i].len;
  }

  char *buf_h, *buf_h_itr; // pinned memory
  size_t *indices_h;       // pinned memory
  score_t *res_scores_h;   // pinned memory
  char *buf_d, *pattern_d;
  size_t *indices_d;
  score_t *res_scores_d;
  size_t prev_sum, n_threads;

  cudaMallocHost(&buf_h, buf_size * sizeof(char));
  cudaMallocHost(&indices_h, matches->count * sizeof(size_t));
  cudaMallocHost(&res_scores_h, matches->count * sizeof(score_t));

  cudaMalloc(&buf_d, buf_size * sizeof(char));
  cudaMalloc(&indices_d, matches->count * sizeof(size_t));
  cudaMalloc(&pattern_d, pattern.len * sizeof(char));
  cudaMalloc(&res_scores_d, matches->count * sizeof(score_t));

  buf_h_itr = buf_h;
  prev_sum = 0, n_threads = 0;
  for (size_t i = 0; i < matches->count; ++i) {
    memcpy(buf_h_itr, matches->items[i].data,
           matches->items[i].len * sizeof(char));
    buf_h_itr += matches->items[i].len;

    indices_h[i] = prev_sum;
    prev_sum += matches->items[i].len;

    n_threads =
        matches->items[i].len > n_threads ? matches->items[i].len : n_threads;
  }
  n_threads = 1024 < n_threads ? 1024 : n_threads;

  cudaMemcpy(buf_d, buf_h, buf_size * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(indices_d, indices_h, matches->count * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(pattern_d, pattern.data, pattern.len * sizeof(char),
             cudaMemcpyHostToDevice);

  dim3 numThreads(n_threads);
  dim3 numBlocks(matches->count);
  fused_score_kernel<<<numBlocks, numThreads>>>(buf_d, indices_d, buf_size,
                                                matches->count, pattern_d,
                                                pattern.len, res_scores_d);

  cudaMemcpy(res_scores_h, res_scores_d, matches->count * sizeof(score_t),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < matches->count; ++i) {
    scored_entry_t s = {.str = matches->items[i], .score = res_scores_h[i]};
    da_append(res, s, scored_entry_t);
  }

  cudaFree(buf_h);
  cudaFree(indices_h);
  cudaFree(res_scores_h);

  cudaFree(buf_d);
  cudaFree(indices_d);
  cudaFree(pattern_d);
  cudaFree(res_scores_d);

  return res;
}
