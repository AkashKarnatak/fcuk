#include "fcuk.h"

__global__ void score_kernel(const char *__restrict__ str,
                             const char *__restrict__ pattern,
                             score_t *match_bonus, score_t *M, score_t *D,
                             size_t n_str, size_t n_ptrn) {
  size_t j = threadIdx.x;

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
                            : match_bonus[j - 1];

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

  n_str = strlen(str);
  n_ptrn = strlen(pattern);

  assert(n_str <= 1024 && n_ptrn <= 1024);

  score_t match_bonus[n_str];
  compute_bonus(str, n_str, match_bonus);

  char *str_d, *pattern_d;
  score_t res, *M_d, *D_d, *match_bonus_d;
  score_t *M;
  // score_t M[n_str + 1][n_ptrn + 1];

  M = (score_t *)malloc((n_ptrn + 1) * (n_str + 1) * sizeof(score_t));

  cudaMalloc(&str_d, n_str * sizeof(char));
  cudaMalloc(&pattern_d, n_ptrn * sizeof(char));
  cudaMalloc(&M_d, (n_ptrn + 1) * (n_str + 1) * sizeof(score_t));
  cudaMalloc(&D_d, (n_ptrn + 1) * (n_str + 1) * sizeof(score_t));
  cudaMalloc(&match_bonus_d, n_str * sizeof(score_t));

  cudaMemcpy(str_d, str, n_str * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(pattern_d, pattern, n_ptrn * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(match_bonus_d, match_bonus, n_str * sizeof(score_t),
             cudaMemcpyHostToDevice);

  dim3 numThreads(n_str + 1);
  dim3 numBlocks(1);

  score_kernel<<<numBlocks, numThreads>>>(str_d, pattern_d, match_bonus_d, M_d,
                                          D_d, n_str, n_ptrn);

  cudaMemcpy(M, M_d, (n_str + 1) * (n_ptrn + 1) * sizeof(score_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&res, &M_d[n_ptrn * (n_str + 1) + n_str], sizeof(score_t),
             cudaMemcpyDeviceToHost);

  cudaFree(str_d);
  cudaFree(pattern_d);
  cudaFree(M_d);
  cudaFree(D_d);
  cudaFree(match_bonus_d);

  return res;
}
