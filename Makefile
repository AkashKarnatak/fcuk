all:
	gcc -g -fsanitize=address -o main.out main.c
