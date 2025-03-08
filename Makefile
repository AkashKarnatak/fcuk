all:
	gcc -g -fsanitize=address -lm fcuk.c main.c -o main.out
