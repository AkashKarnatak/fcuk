# Fcuk - Fuzzy CUDA kernel

fcuk (Fuzzy CUDA Kernel) is a fuzzy matching library implemented in CUDA.
It leverages dynamic programming, wavefront parallelization and parallel
reduction techniques for efficient string matching and scoring.

# Features

- Fuzzy string matching with CUDA acceleration
- Scoring algorithm based on affine gap penalty (similar to fzy)
- Optimized memory usage with pinned and shared memory
- Parallel reduction for matching with O(n log m) time complexity
- Fused kernels for improved performance

# Installation

```
# Clone the repository
git clone https://github.com/AkashKarnatak/fcuk.git
cd fcuk

# Build the project
make
```

# Contributing

Contributions are welcome! If you find a bug, have an idea for an enhancement, or want to contribute in any way, feel free to open an issue or submit a pull request.

# License

This project is licensed under the AGPL3 License. For details, see the [LICENSE](LICENSE) file.
