#include "knn.h"

int main(int argc, char** argv){
    const int n = 5000 - 1;
    const int m = 2;
    const int d = 2;
    const int k = 3;

    double X[] = {200000.0, 200000.0,
                  600000.0, 200000.0};

    double Y[] = {2.0, 4.8,
                  2.0, 1.0,
                  4.0, 4.0,
                  5.0, 5.0,
                  5.0, 1.0,
                  6.0, 3.0,
                  7.0, 2.0};

    distrAllkNN(X, n, d, k);
}