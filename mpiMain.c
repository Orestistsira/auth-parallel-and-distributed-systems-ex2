#include "knn.h"
#include <mpi.h>

int main(int argc, char** argv){
    int SelfTID, p;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    const int n = 8 / p;   
    const int d = 2;
    const int k = 4;

    //init all corpus points
    //double* X = getArrayFromTxt("s1.txt", n * p, d);

    double Xall[] = {2.0, 4.8,
                     2.0, 1.0,
                     4.0, 4.0,
                     5.0, 5.0,
                     5.0, 1.0,
                     6.0, 3.0,
                     7.0, 2.0,
                     3.0, 3.5};

    if(SelfTID == 0){
        printf("Num of tasks: %d\n", p);
        printf("Array length for each task: %d\n", n);
        printArrayDouble(Xall, n * p * d);
    }

    double* X = (double *) malloc(n * d * sizeof(double));

    //Distribute Xall to each task
    MPI_Scatter(Xall, n * d, MPI_DOUBLE, X, n * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printf("X of Task %d:\n", SelfTID);
    printArrayDouble(X, n * d);
    
    //Get knn result from each process
    knnresult knn = distrAllkNN(X, n, d, k);

    sleep(1);

    knnresult knnAll;
    knnAll.ndist = NULL;
    knnAll.nidx = NULL;

    if(SelfTID == 0){
        int knnSize = n * p * k;
        knnAll.ndist = (double*) malloc(knnSize * sizeof(double));
        knnAll.nidx = (int*) malloc(knnSize * sizeof(int));
        knnAll.k = k;
        knnAll.m = n * p;
    }

    //Gather all data from each process to task 0
    MPI_Gather(knn.ndist, n * k, MPI_DOUBLE, knnAll.ndist, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(knn.nidx, n * k, MPI_INT, knnAll.nidx, n * k, MPI_INT, 0, MPI_COMM_WORLD);

    if(SelfTID == 0){
        printArrayDouble(knnAll.ndist, n * p * k);
        printArrayInt(knnAll.nidx, n * p * k);
    } 

    MPI_Finalize();

    free(knn.ndist);
    free(knn.nidx);
    free(knnAll.ndist);
    free(knnAll.nidx);

    return 0;
}