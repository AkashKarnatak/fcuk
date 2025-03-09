#include "fcuk.h"

__device__ bool islower(char c) { return c >= 'a' && c <= 'z'; }

__device__ bool isupper(char c) { return c >= 'A' && c <= 'Z'; }

__device__ bool isdigit(char c) { return c >= '0' && c <= '9'; }

__device__ bool isspecial(char c) {
  return c == '/' || c == '-' || c == '_' || c == ' ' || c == '.';
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
        M[i * (n_str + 1)] = 0;
        continue;
      }

      score_t gap_penalty =
          i == n_ptrn ? GAP_PENALTY_TRAILING : GAP_PENALTY_INNER;

      if (str[j - 1] == pattern[i - 1]) {
        score_t bonus = (D[(i - 1) * (n_str + 1) + j - 1] != SCORE_MIN)
                            ? CONSECUTIVE_BONUS
                            : match_bonus_s[j - 1];

        D[i * (n_str + 1) + j] = M[(i - 1) * (n_str + 1) + j - 1] + bonus;
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

score_t score(const char *__restrict__ str, const char *__restrict__ pattern) {
  size_t n_str, n_ptrn;
  char *str_d, *pattern_d;
  score_t res, *M_d, *D_d;

  n_str = strlen(str);
  n_ptrn = strlen(pattern);

  assert(n_str <= 1024 && n_ptrn <= 1024);

  // TODO: figure out a better way
  if (!init) {
    cudaMemcpyToSymbol(SPECIAL_BONUS_C, SPECIAL_BONUS, sizeof(SPECIAL_BONUS));
    init = true;
  }

  cudaMalloc(&str_d, n_str * sizeof(char));
  cudaMalloc(&pattern_d, n_ptrn * sizeof(char));
  cudaMalloc(&M_d, (n_ptrn + 1) * (n_str + 1) * sizeof(score_t));
  cudaMalloc(&D_d, (n_ptrn + 1) * (n_str + 1) * sizeof(score_t));

  cudaMemcpy(str_d, str, n_str * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(pattern_d, pattern, n_ptrn * sizeof(char), cudaMemcpyHostToDevice);

  dim3 numThreads(n_str + 1);
  dim3 numBlocks(1);
  fused_score_kernel<<<numBlocks, numThreads>>>(str_d, pattern_d, M_d, D_d,
                                                n_str, n_ptrn);

  cudaMemcpy(&res, &M_d[n_ptrn * (n_str + 1) + n_str], sizeof(score_t),
             cudaMemcpyDeviceToHost);

  cudaFree(str_d);
  cudaFree(pattern_d);
  cudaFree(M_d);
  cudaFree(D_d);

  return res;
}
