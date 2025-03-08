all:
	gcc -Wall -Werror -g -O3 -fsanitize=address -lm common.c fcuk.c main.c -o main.out
	nvcc -O3 -Xcompiler -Wall -Xcompiler -Werror -arch=sm_86 -lm common.c fcuk.cu main.c -o main.cu.out
