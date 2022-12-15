#include "knn.h"

int main(int argc, char** argv){

    const int n = 2;
    const int m = 2;
    const int d = 3;
    const int k = 1;

    double X[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};

    double Y[] = {1.0, 5.0, 2.0,
                  3.0, 2.0, 4.0};

    printArrayDouble(X, m * d);
    printArrayDouble(Y, n * d);
    printf("\n");

    knnresult knn = kNN(X, Y, n, m, d, k);

    free(knn.ndist);
    free(knn.nidx);
    
    return 0;
}