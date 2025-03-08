#include "fcuk.h"

#define da_append(xs, x)                                                       \
  do {                                                                         \
    if (xs.count >= xs.capacity) {                                             \
      if (xs.capacity == 0)                                                    \
        xs.capacity = 256;                                                     \
      else                                                                     \
        xs.capacity *= 2;                                                      \
      xs.items = realloc(xs.items, xs.capacity * sizeof(*xs.items));           \
    }                                                                          \
    xs.items[xs.count++] = x;                                                  \
  } while (0)

typedef struct {
  char **items;
  size_t count;
  size_t capacity;
} Strings;

int main(int argc, char *argv[]) {
  char *buf, *pattern;
  size_t buf_size;
  int32_t n;
  Strings strings = {0};

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <PATTERN>\n", argv[0]);
    exit(1);
  }

  pattern = argv[1];

  buf = NULL;
  while ((n = getline(&buf, &buf_size, stdin)) != -1) {
    buf[n - 1] = '\0';
    da_append(strings, buf);
    buf = NULL;
  }
  free(buf);

  for (size_t i = 0; i < strings.count; ++i) {
    if (match(strings.items[i], pattern)) {
      printf("%s: %lf\n", strings.items[i], score(strings.items[i], pattern));
    }
  }

  // clean up
  for (size_t i = 0; i < strings.count; ++i) {
    free(strings.items[i]);
  }
  free(strings.items);
}
