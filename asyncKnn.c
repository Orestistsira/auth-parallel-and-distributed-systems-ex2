#include "knn.h"
#include <mpi.h>

knnresult distrAllkNN(double* X, int n, int d, int k){
    int SelfTID, NumTasks, t, data;
    MPI_Status mpistat;

    MPI_Init(NULL, NULL);
    MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
    MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );
    printf("Hello World from %i of %i\n", SelfTID,NumTasks);

    if( SelfTID == 0 ) {
        for(t=1;t<NumTasks;t++) {
            data = t;
            MPI_Send(&data,1,MPI_INT,t,55,MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&data,1,MPI_INT,0,55,MPI_COMM_WORLD,&mpistat);
        printf("TID%i: received data=%i\n",SelfTID,data);
    }

    MPI_Finalize();

    knnresult knn;
    return knn;
}