#include "knn.h"
#include <mpi.h>
#include <unistd.h>
#include <float.h>

knnresult distrAllkNN(double* X, int n, int d, int k){
    int SelfTID, p, err;
    MPI_Status mpistat;
    MPI_Request mpireq;

    MPI_Comm_size( MPI_COMM_WORLD, &p );
    MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );

    knnresult knnAll;
    int knnSize = n * p * k;
    knnAll.ndist = (double*) malloc(knnSize * sizeof(double));
    knnAll.nidx = (int*) malloc(knnSize * sizeof(int));

    //Y = X for the first iteration
    double* Y = copyArray(X, n * d);
    //Allocate Z for the incoming messages
    double* Z = (double *) malloc(n * d * sizeof(double));

    sleep(1);

    printf("Y of Task %d:\n", SelfTID);
    printArrayDouble(Y, n * d);

    for(int pass=0;pass<1;pass++){
        //No need to pass Y to the starting point
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

        //Calculate knn for each task
        sleep(1);

        knnresult knn = kNN(X, Y, n, n, d, k);

        //Change point ids to global ids
        for(int i=0;i<n;i++){
            for(int j=0;j<k;j++){
                int mult = (SelfTID - pass - 1) < 0  ? p + SelfTID - pass - 1 : SelfTID - pass - 1;
                knn.nidx[i * k + j] += mult * n;
            }
        }

        printf("Task %d knn ", SelfTID);
        printf("result:\n");
        printArrayDouble(knn.ndist, k * n);
        printArrayInt(knn.nidx, k * n);
        printf("\n");

        sleep(1);

        //TODO: Calculate knnAll
        for(int i=0;i<n;i++){
            //for current point i, check if there is a shorter distance in knn than those in knnAll
            for(int j=0;j<k;j++){
                knnAll.ndist[i * k + j] = knn.ndist[i * k + j];
                knnAll.nidx[i * k + j] = knn.nidx[i * k + j];
            }
        }

        //Receive Z and put it to Y
        if(pass < p - 1) MPI_Wait(&mpireq, &mpistat);
        //free(Y);
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
    

    MPI_Finalize();

    free(Z);

    return knnAll;
}