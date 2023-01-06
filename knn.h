#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cblas.h>
#include <omp.h>
#include <unistd.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>

typedef struct knnresult{
    int* nidx;
    double* ndist;
    int m;
    int k;
} knnresult;

double* getArrayFromTxt(char* filename, int numOfpoints, int dimension, int startingRow, int endingRow);

double* copyArray(double const* src, int len);

void printArrayDouble(double* arr, int size);

void printArrayInt(int* arr, int size);

void quickSort(double* array, int* otherArray, int low, int high, int k);

knnresult kNN(double* X, double* Y, int n, int m, int d, int k);

knnresult distrAllkNN(double* X, int n, int d, int k);