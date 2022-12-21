#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cblas.h>
#include <omp.h>

typedef struct knnresult{
    int* nidx;
    double* ndist;
    int m;
    int k;
} knnresult;

void printArrayDouble(double* arr, int size);

knnresult kNN(double* X, double* Y, int n, int m, int d, int k);