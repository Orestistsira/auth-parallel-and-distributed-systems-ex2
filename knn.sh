#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=1:00:00

module load gcc openblas

gcc main.c knn.c -lm -I$OPENBLAS_ROOT/include -L$OPENBLAS_ROOT/lib -lopenblas -O3 -o knn.out

srun ./knn.out $1 $2 $3 $4
