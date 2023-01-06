# auth-parallel-and-distributed-systems-ex2

sudo apt-get install libopenblas-dev

make knn

./knn.out [filepath] [n] [d] [k] [print]

E.g. ./knn.out ./data/regular3d.txt 27 3 27 print

make knn-async

mpiexec -n 4 ./knn-async.out

mpirun -np 4 ./knn-async.out