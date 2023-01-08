#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --time=1:00:00

module load gcc openmpi openblas

mpicc mpiMain.c asyncKnn.c knn.c -lm -I$OPENBLAS_ROOT/include -L$OPENBLAS_ROOT/lib -lopenblas -O3 -o knn-async.out

srun ./knn-async.out $1 $2 $3 $4
