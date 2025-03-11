#include "fcuk.h"

int main(int argc, char *argv[]) {
  char *buf;
  string_t pattern;
  size_t buf_size;
  int32_t n;
  strings_t entries = {0};
  strings_t matches = {0};
  results_t res = {0};

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <PATTERN>\n", argv[0]);
    exit(1);
  }

  pattern = (string_t){.data = argv[1], .len = strlen(argv[1])};

  buf = NULL;
  while ((n = getline(&buf, &buf_size, stdin)) != -1) {
    buf[n - 1] = '\0';
    string_t s = {.data = buf, .len = n - 1};
    da_append(entries, s, string_t);
    buf = NULL;
  }
  free(buf);

  matches = match(&entries, pattern);
  res = score_matches(&matches, pattern);

  for (size_t i = 0; i < res.count; ++i) {
    printf("%s: %lf\n", res.items[i].str.data, res.items[i].score);
  }

  // clean up
  for (size_t i = 0; i < entries.count; ++i) {
    free(entries.items[i].data);
  }
  free(entries.items);
  free(matches.items);
  free(res.items);
}
