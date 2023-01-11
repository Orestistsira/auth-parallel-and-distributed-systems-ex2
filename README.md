# auth-parallel-and-distributed-systems-ex2

Compilers:

gcc (11.3.0)

---

If you don't have openblas installed:

```

sudo apt-get install libopenblas-dev

```

---

(print writes the output to a file, it can be skipped for all commands)

1. To run sequential code

```

make knn

./knn.out [filepath] [n] [d] [k] [print]

```

E.g. ./knn.out ./data/regular3d.txt 27 3 27 print

1. To run mpi asynchronus code

```

make knn-async

mpirun -np 4 ./knn-async.out [filepath] [n] [d] [k] [print]

```

or

```

mpiexec -n 4 ./knn-async.out [filepath] [n] [d] [k] [print]

```
