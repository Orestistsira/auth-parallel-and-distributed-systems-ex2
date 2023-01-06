#include "knn.h"
#include "arrayMaker.h"
#include <mpi.h>

int main(int argc, char** argv){
    int SelfTID, p;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    srand(time(NULL) + SelfTID);

    int numOfPoints, n, d, k;

    char* filepath;
    char* pr;
    bool print = false;

    char substring[6];
    
    if(argc >= 5){
        filepath = argv[1];
        numOfPoints = atoi(argv[2]);
        n = (numOfPoints + p - 1) / p;
        d = atoi(argv[3]);
        k = atoi(argv[4]);

        pr = argv[5];
        if(argc == 6 && !strcmp("print", pr)){
            print = true;
        }
    }

    if(SelfTID == 0){
        printf("n = %d\n", n);
        printf("Loading arrays...\n");
    } 

    double* X;
    double* Xall;
    strcpy(substring, "ubyte");

    if(!strcmp("random", filepath)){
        X = getRandomArray(n, d);
    }
    else if(strstr(filepath, substring) != NULL){
        X = getMinstArray(filepath, SelfTID * n, (SelfTID + 1) * n);
    }
    else{
        X = getArrayFromTxt(filepath, n * p, d, SelfTID * n, (SelfTID + 1) * n);
    }

    double starttime, endtime;
    double duration;
    
    if(SelfTID == 0){
        printf("Num of tasks: %d\n", p);
        printf("Array length for each task: %d\n", n * d);
        printf("---------------------------------------------\n");
        //printArrayDouble(Xall, n * p * d);
    }
    starttime = MPI_Wtime();

    // printf("X of Task %d:\n", SelfTID);
    //printArrayDouble(X, n * d);
    
    //Get knn result from each process
    knnresult knn = distrAllkNN(X, n, d, k);

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

    endtime = MPI_Wtime();

    duration = endtime - starttime;
    printf("[Async kNN took %f seconds for task %d]\n", duration, SelfTID);

    if(SelfTID == 0 && print){
        printf("\nEnd result:\n");
        printResult(&knnAll, numOfPoints);
    }

    MPI_Finalize();

    free(knn.ndist);
    free(knn.nidx);
    free(knnAll.ndist);
    free(knnAll.nidx);

    return 0;
}