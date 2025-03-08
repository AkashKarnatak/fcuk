all:
	gcc -Wall -Werror -g -O3 -fsanitize=address -lm common.c fcuk.c main.c -o main.out
