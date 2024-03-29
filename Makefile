SHELL := /bin/bash

CC       = gcc
FLAGS    = -lm -lopenblas -O3 -g

CC_MPI   = mpicc
FLAGS_MPI= -lm -lopenblas -O3 -g

knn: main.c knn.c
	$(CC) main.c knn.c $(FLAGS) -o knn.out

knn-async: mpiMain.c asyncKnn.c
	$(CC_MPI) mpiMain.c asyncKnn.c knn.c $(FLAGS_MPI) -o knn-async.out

tester: regularGridTester.c
	$(CC) regularGridTester.c -o tester.out

clean:
	$(RM) knn.out