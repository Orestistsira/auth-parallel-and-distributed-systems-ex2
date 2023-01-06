#include "knn.h"
#include "arrayMaker.h"

int main(int argc, char** argv){
    int n, m, d, k;
    char* filepath;
    char* pr;
    bool print = false;

    char substring[6];
    
    if(argc >= 5){
        filepath = argv[1];
        n = atoi(argv[2]);
        m = n;
        d = atoi(argv[3]);
        k = atoi(argv[4]);

        strcpy(substring, "print");
        pr = argv[5];
        if(argc == 6 && strstr(pr, substring) != NULL){
            print = true;
        }
    }

    printf("Loading arrays...\n");

    double* Y;
    double* X;
    strcpy(substring, "ubyte");

    if(!strcmp("random", filepath)){
        Y = getRandomArray(n, d);
    }
    else if(strstr(filepath, substring) != NULL){
        Y = getMinstArray(filepath, 0, n);
    }
    else{
        Y = getArrayFromTxt(filepath, n, d, 0, n);
    }
    X = copyArray(Y, n * d);

    struct timeval startwtime, endwtime;
    double duration;

    printf("Calculating kNN...\n");

    gettimeofday (&startwtime, NULL);
    knnresult knn = kNN(X, Y, n, m, d, k);
    gettimeofday (&endwtime, NULL);

    duration = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    printf("[kNN took %f seconds]\n", duration);

    if(print){
        int knnSize = k * m;
        if(!strcmp("random", filepath))
            printPoints(Y, n, d);
        printResult(&knn);
    }

    free(knn.ndist);
    free(knn.nidx);
    free(X);
    free(Y);

    return 0;
}