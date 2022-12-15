#include "knn.h"

#define DEBUG_INT(num) printf("%d\n", num)
#define DEBUG_DOUBLE(num) printf("%f\n", num)
#define DEBUG_STR(str) printf("%s\n", str)
#define ERROR(msg) fprintf(stderr, "Error: %s\n", msg)

void printArrayDouble(double* arr, int size){
    for(int i=0;i<size;i++){
        printf("%f ", arr[i]);
    }
    printf("\n");
}

void printArrayInt(int* arr, int size){
    for(int i=0;i<size;i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int* copyArray(int const* src, int len){
	int* p = malloc(len * sizeof(int));
	memcpy(p, src, len * sizeof(int));
	return p;
}

void swapDouble(double *a, double *b){
	double t = *a;
	*a = *b;
	*b = t;
}

void swapInt(int *a, int *b){
	int t = *a;
	*a = *b;
	*b = t;
}

int partition(double* array, int* otherArray, int low, int high){
  
	// select the rightmost element as pivot
	double pivot = array[high];
	
	// pointer for greater element
	int i = (low - 1);

	// traverse each element of the array
	// compare them with the pivot
	for(int j = low; j < high; j++){
		if(array[j] <= pivot){
            // if element smaller than pivot is found
            // swap it with the greater element pointed by i
            i++;
            
            // swap element at i with element at j
            swapDouble(&array[i], &array[j]);
            swapInt(&otherArray[i], &otherArray[j]);
		}
	}

	// swap the pivot element with the greater element at i
	swapDouble(&array[i + 1], &array[high]);
	swapInt(&otherArray[i + 1], &otherArray[high]);
	
	// return the partition point
	return (i + 1);
}

void quickSort(double* array, int* otherArray, int low, int high){
	if (low < high) {
		
		// find the pivot element such that
		// elements smaller than pivot are on left of pivot
		// elements greater than pivot are on right of pivot
		int pi = partition(array, otherArray, low, high);
		
		// recursive call on the left of pivot
		quickSort(array, otherArray, low, pi - 1);
		
		// recursive call on the right of pivot
		quickSort(array, otherArray, pi + 1, high);
	}
}

void calculateDistances(double* D, double* X, double* Y, int m, int n, int d){
    const int xSize = m * d;
    const int ySize = n * d;

    int xIndex = 0;
    for(int x=0;x<xSize;x+=d){
        int yIndex = 0;
        for(int y=0;y<ySize;y+=d){
            double sum = 0;

            for(int i=0;i<d;i++){
                sum += pow(X[x + i] - Y[y + i], 2);
            }

            D[xIndex * m + yIndex] = sqrt(sum);

            yIndex++;
        }

        xIndex++;
    }
}

knnresult kNN(double* X, double* Y, int n, int m, int d, int k){
    if(k > n){
        ERROR("k can't be greater than n");
        exit(1);
    }

    struct timeval startwtime, endwtime;
    double duration;

    //Init knn
    knnresult knn;
    knn.k = k;
    knn.m = m;

    int knnSize = k * m;
    knn.ndist = (double*) malloc(knnSize * sizeof(double));
    knn.nidx = (int*) malloc(knnSize * sizeof(int));

    const int xSize = m * d;
    const int ySize = n * d;

    //Init distances O(m x n x d^3)
    int distancesSize = n * m;
    double* D = (double *) malloc(distancesSize * sizeof(double));
    
    gettimeofday (&startwtime, NULL);
    calculateDistances(D, X, Y, m, n, d);
    gettimeofday (&endwtime, NULL);

    duration = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("[D took %f seconds]\n", duration);

    printf("D: ");
    printArrayDouble(D, distancesSize);

    gettimeofday (&startwtime, NULL);
    
    //Init distances O(m x n x d)
    //calculate sum(X.^2, 2) and put it to A
    double* A = (double *) malloc(m * sizeof(double));

    for(int i=0;i<m;i++){
        double sumA = 0;
        for(int dim=0;dim<d;dim++){
            sumA += pow(X[i * d + dim], 2);
        }
        A[i] = sumA;
    }

    //calculate sum(Y.^2, 2).' and put it to C
    //C is transposed!!
    double* C = (double *) malloc(n * sizeof(double));

    for(int j=0;j<n;j++){
        double sumB = 0;
        for(int dim=0;dim<d;dim++){
            sumB += pow(Y[j * d + dim], 2);
        }
        C[j] = sumB;
    }

    //calculate  - 2 * X*Y.' and put it to B
    double* B = (double *) malloc(m * n * sizeof(double));

    // void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
    //              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
    //              const int K, const double alpha, const double *A,
    //              const int lda, const double *B, const int ldb,
    //              const double beta, double *C, const int ldc);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2, X, d, Y, d, 0, B, n);

    // printArrayDouble(B, distancesSize);

    //Calculate D
    for(int i=0;i<m;i++){
        //A can be done here
        for(int j=0;j<n;j++){
            //C can be done here
            D[i * n + j] = sqrt(A[i] + B[i * n + j] + C[j]);
        }
    }

    free(A);
    free(B);
    free(C);

    gettimeofday (&endwtime, NULL);

    duration = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("[D took %f seconds]\n", duration);

    printf("D: ");
    printArrayDouble(D, distancesSize);

    //Init yids O(n * m)
    int* yId = (int*) malloc(distancesSize * sizeof(int));

    for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			yId[i * n + j] = j;
		}
    }
    
    // printf("yId: ");
    // printArrayInt(yId, distancesSize);
    DEBUG_STR("Calculating knn:");
    DEBUG_STR("------------------------------------------------------");

    //calculate knn
    for(int i=0;i<m;i++){
        printf("D: ");
        printArrayDouble(D, distancesSize);
        printf("yId: ");
        printArrayInt(yId, distancesSize);

		//Quicksort D array and move id elements the same way
        quickSort(D, yId, i * n, i * n + n - 1);

        printf("D sorted: ");
        printArrayDouble(D, distancesSize);
        printf("yId sorted: ");
        printArrayInt(yId, distancesSize);

        for(int j=0;j<k;j++){
            knn.ndist[i * k + j] = D[i * n + j];
            knn.nidx[i * k + j] = yId[i * n + j];
        }
    }

    printf("\n");
    printf("result:\n");
    printArrayDouble(knn.ndist, knnSize);
    printArrayInt(knn.nidx, knnSize);

    printf("\n");
    printArrayDouble(X, m * d);
    printArrayDouble(Y, n * d);

    free(yId);
    free(D);

    return knn;
}