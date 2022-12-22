SHELL := /bin/bash

CC       = gcc
FLAGS    = -lm -lopenblas -fopenmp -O3

CC_MPI   = mpicc
FLAGS_MPI= -lm -lopenblas -fopenmp -O3

knn: main.c knn.c
	$(CC) main.c knn.c $(FLAGS) -o knn.out

run:
	./knn.out

knn-async: mpiMain.c asyncKnn.c
	$(CC_MPI) mpiMain.c asyncKnn.c knn.c $(FLAGS_MPI) -o knn-async.out

clean:
	$(RM) colorScc