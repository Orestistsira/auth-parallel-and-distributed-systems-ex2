#include "knn.h"
#include <mpi.h>

knnresult distrAllkNN(double* X, int n, int d, int k){
    int SelfTID, p, err;
    MPI_Status mpistat;
    MPI_Request mpireqSend, mpireqRecv;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

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

    // printf("Y of Task %d:\n", SelfTID);
    // printArrayDouble(Y, n * d);

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
            err = MPI_Isend(Y, n * d, MPI_DOUBLE, destTaskId, tag, MPI_COMM_WORLD, &mpireqSend);
            if(err){
                printf("Error=%i in MPI_Isend to %i\n", err, destTaskId);
            }

            int sourcetaskId = (SelfTID - 1) == -1 ? p - 1 : SelfTID - 1;
            MPI_Irecv(Z, n * d, MPI_DOUBLE, sourcetaskId, tag, MPI_COMM_WORLD, &mpireqRecv);
        }

        //Calculate knn for each task
        knnresult knn = kNN(X, Y, n, n, d, k);

        //Change point ids to global ids
        for(int i=0;i<n;i++){
            for(int j=0;j<k;j++){
                int mult = (SelfTID - pass) < 0  ? p + SelfTID - pass : SelfTID - pass;
                knn.nidx[i * k + j] += mult * n;
            }
        }

        // printf("Task %d knn ", SelfTID);
        // printf("result:\n");
        // printArrayDouble(knn.ndist, k * n);
        // printArrayInt(knn.nidx, k * n);
        // printf("\n");

        int tempIndex = 0;
        //Calculate knnAll
        //for current point i, check if there is a shorter distance in knn than those in knnAll
        for(int i=0;i<n;i++){
            //if the first element of the current distance array is larger than the last element of the global distance array continue
            //as there is no shorter distances than already calculated
            if(knn.ndist[i * k] >= knnAll.ndist[i * k + k - 1]) continue;

            tempIndex = 0;
            for(int j=0;j<k;j++){
                ndistTemp[tempIndex] = knnAll.ndist[i * k + j];
                nidxTemp[tempIndex++] = knnAll.nidx[i * k + j];

                ndistTemp[tempIndex] = knn.ndist[i * k + j];
                nidxTemp[tempIndex++] = knn.nidx[i * k + j];
            }
            // printf("Task %d temp:\n", SelfTID);
            // printArrayDouble(ndistTemp, 2 * k);

            quickSort(ndistTemp, nidxTemp, 0, 2 * k - 1, k);

            for(int j=0;j<k;j++){
                knnAll.ndist[i * k + j] = ndistTemp[j];
                knnAll.nidx[i * k + j] = nidxTemp[j];
            }
        }

        // printf("Task %d knn ", SelfTID);
        // printf("result all:\n");
        // printArrayDouble(knnAll.ndist, k * n);
        // printArrayInt(knnAll.nidx, k * n);
        // printf("\n");

        
        if(pass < p - 1){
            MPI_Wait(&mpireqSend, &mpistat);
            MPI_Wait(&mpireqRecv, &mpistat);
        }
        
        //Receive Z and put it to Y
        double* temp = Y;
        Y = Z;
        Z = temp;
        //DONE?: use pointers, do not copy array
        //Y = copyArray(Z, n * d);

        // printf("Y of Task %d changed:\n", SelfTID);
        // printArrayDouble(Y, n * d);

        free(knn.ndist);
        free(knn.nidx);
    }

    // printf("\n");
    // printf("Task %d knn ", SelfTID);
    // printf("result all:\n");
    // printArrayDouble(knnAll.ndist, k * n);
    // printArrayInt(knnAll.nidx, k * n);
    // printf("\n");
    
    free(X);
    free(Y);
    free(Z);
    free(ndistTemp);
    free(nidxTemp);

    return knnAll;
}