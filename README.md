# auth-parallel-and-distributed-systems-ex2

**k Nearest Neighbors**

---

Compilers:

gcc (11.3.0)

---

If you don't have openblas installed on Linux:

```

sudo apt-get install libopenblas-dev

```

---

Compile and Run the code

- filepath = path to txt/mnist file (random can be put to execute code for random dataset)

- n = number of points

- d = dimensions

- k = number of neighbors

- print writes the output to a file, it can be skipped for all commands

1. To run sequential code

```

make knn

./knn.out [filepath] [n] [d] [k] [print]

```

E.g. ./knn.out ./data/regular3d.txt 27 3 27 print

2. To run mpi asynchronus code

```

make knn-async

mpirun -np 4 ./knn-async.out [filepath] [n] [d] [k] [print]

```

or

```

mpiexec -n 4 ./knn-async.out [filepath] [n] [d] [k] [print]

```
---

Run code on HPC get scripts from /script folder

1. To run sequential code

```

sbatch knn.sh [filepath] [n] [d] [k] [print]

```

2. To run mpi asynchronus code on 4 nodes

```

sbatch knn-async.sh [filepath] [n] [d] [k] [print]

```

---

To check if the regular grid file outputs of sequential code are correct

```

make tester

./tester [n] [d] [k]

```

*The tester is very simple as i did not have enough time*
