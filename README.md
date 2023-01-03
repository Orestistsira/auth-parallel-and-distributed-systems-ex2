# auth-parallel-and-distributed-systems-ex2

sudo apt-get install libopenblas-dev

make knn

make run

make knn-async

mpiexec -n 4 ./knn-async.out

mpirun -np 4 ./knn-async.out