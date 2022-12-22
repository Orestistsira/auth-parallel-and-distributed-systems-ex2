#include "knn.h"
#include <mpi.h>

int main(int argc, char** argv){
    int SelfTID, p, err;
    MPI_Status mpistat;

    MPI_Init(&argc, &argv);

    MPI_Comm_size( MPI_COMM_WORLD, &p );
    MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );

    const int n = 8 / p;      
    const int d = 2;
    const int k = 2;

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
    double* Xtemp = (double *) malloc(n * d * sizeof(double));

    //task id=0 keeps the last part of Xall
    if(SelfTID == 0){
        for(int id=0;id<p;id++){

            for(int i=0;i<n;i++){
                for(int j=0;j<d;j++){
                    X[i * d + j] = Xall[id * n * d + i * d + j];
                }
            }

            if(id == p - 1) break;

            int destTaskId = id + 1;
            int tag = 1;
            err = MPI_Send(X, n * d, MPI_DOUBLE, destTaskId, tag, MPI_COMM_WORLD);

            if(err){
                printf("Error=%i in MPI_Send to %i\n", err, destTaskId);
            }
        }
    }
    else{
        MPI_Recv(X, n * d, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &mpistat);
    }

    printf("X of Task %d:\n", SelfTID);
    printArrayDouble(X, n * d);
        
    knnresult knn = distrAllkNN(X, n, d, k);

    free(X);

    return 0;
}