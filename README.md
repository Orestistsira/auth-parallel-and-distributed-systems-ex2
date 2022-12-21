# auth-parallel-and-distributed-systems-ex2

sudo apt-get install libopenblas-dev

gcc main.c knn.c -lm -lopenblas -fopenmp -O3 -o knn.out

mpicc  main.c asyncKnn.c -lm -lopenblas -fopenmp -O3 -o knn-async.out

mpiexec -n 4 ./knn-async.out