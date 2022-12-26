#include "knn.h"
#include <mpi.h>
#include <float.h>

knnresult distrAllkNN(double* X, int n, int d, int k){
    int SelfTID, p, err;
    MPI_Status mpistat;
    MPI_Request mpireq;

    MPI_Comm_size( MPI_COMM_WORLD, &p );
    MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );

    knnresult knnAll;
    int knnSize = n * k;
    knnAll.ndist = (double*) malloc(knnSize * sizeof(double));
    knnAll.nidx = (int*) malloc(knnSize * sizeof(int));
    knnAll.k = k;
    knnAll.m = n;

    for(int i=0;i<knnSize;i++){
        knnAll.ndist[i] = DBL_MAX;
    }

    //Y = X for the first iteration
    double* Y = copyArray(X, n * d);
    //Allocate Z for the incoming messages
    double* Z = (double *) malloc(n * d * sizeof(double));

    sleep(1);

    printf("Y of Task %d:\n", SelfTID);
    printArrayDouble(Y, n * d);

    //TODO: Do it without temp arrays
    //Allocate temp arrays to compare the results and find the nearest points
    double* ndistTemp = (double*) malloc(2 * k * sizeof(double));
    int* nidxTemp = (int*) malloc(2 * k * sizeof(int));

    for(int pass=0;pass<p;pass++){
        //No need to pass Y back to the starting point
        if(pass < p - 1){
            //Send Y to the next task
            int destTaskId = (SelfTID + 1) % p;
            int tag = 2;
            err = MPI_Isend(Y, n * d, MPI_DOUBLE, destTaskId, tag, MPI_COMM_WORLD, &mpireq);
            if(err){
                printf("Error=%i in MPI_Isend to %i\n", err, destTaskId);
            }

            int sourcetaskId = (SelfTID - 1) == -1 ? p - 1 : SelfTID - 1;
            MPI_Irecv(Z, n * d, MPI_DOUBLE, sourcetaskId, tag, MPI_COMM_WORLD, &mpireq);
        }

        sleep(1);

        //Calculate knn for each task
        knnresult knn = kNN(X, Y, n, n, d, k);

        //Change point ids to global ids
        for(int i=0;i<n;i++){
            for(int j=0;j<k;j++){
                int mult = (SelfTID - pass) < 0  ? p + SelfTID - pass : SelfTID - pass;
                knn.nidx[i * k + j] += mult * n;
            }
        }

        printf("Task %d knn ", SelfTID);
        printf("result:\n");
        printArrayDouble(knn.ndist, k * n);
        printArrayInt(knn.nidx, k * n);
        printf("\n");

        sleep(1);

        //Calculate knnAll
        for(int i=0;i<n;i++){
            //for current point i, check if there is a shorter distance in knn than those in knnAll
            int tempIndex = 0;
            for(int j=0;j<k;j++){
                ndistTemp[tempIndex] = knnAll.ndist[i * k + j];
                nidxTemp[tempIndex++] = knnAll.nidx[i * k + j];

                ndistTemp[tempIndex] = knn.ndist[i * k + j];
                nidxTemp[tempIndex++] = knn.nidx[i * k + j];
            }
            printf("Task %d temp:\n", SelfTID);
            printArrayDouble(ndistTemp, 2 * k);

            quickSort(ndistTemp, nidxTemp, 0, 2 * k - 1);

            for(int j=0;j<k;j++){
                knnAll.ndist[i * k + j] = ndistTemp[j];
                knnAll.nidx[i * k + j] = nidxTemp[j];
            }
        }

        //Receive Z and put it to Y
        if(pass < p - 1) MPI_Wait(&mpireq, &mpistat);
        Y = Z;

        sleep(1);

        printf("Y of Task %d changed:\n", SelfTID);
        printArrayDouble(Y, n * d);
    }

    sleep(1);

    printf("Task %d knn ", SelfTID);
    printf("result all:\n");
    printArrayDouble(knnAll.ndist, k * n);
    printArrayInt(knnAll.nidx, k * n);
    printf("\n");
    
    free(X);
    free(Z);
    free(ndistTemp);
    free(nidxTemp);

    return knnAll;
}