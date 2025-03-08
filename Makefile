all:
	gcc -g -fsanitize=address -lm main.c -o main.out
