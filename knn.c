#include "knn.h"

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

double* copyArray(double const* src, int len){
    double* p = malloc(len * sizeof(double));
    if(p == NULL)
        printf("Error: malloc failed in copy array\n");
    memcpy(p, src, len * sizeof(double));
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

void quickSort(double* array, int* otherArray, int low, int high, int k){
	if (low < high) {
		
		// find the pivot element such that
		// elements smaller than pivot are on left of pivot
		// elements greater than pivot are on right of pivot
		int pi = partition(array, otherArray, low, high);
		
		// recursive call on the left of pivot
		quickSort(array, otherArray, low, pi - 1, k);

        //if the k shortest distances are on the left, there is no need to short the right part
        if(pi - low >= k) return;
		
		// recursive call on the right of pivot
		quickSort(array, otherArray, pi + 1, high, k);
	}
}

knnresult kNN(double* X, double* Y, int n, int m, int d, int k){
    if(k > n){
        printf("Error In kNN: k can't be greater than n\n");
        exit(-1);
    }

    //Init knn
    knnresult knn;
    knn.k = k;
    knn.m = m;

    int knnSize = k * m;
    knn.ndist = (double*) malloc(knnSize * sizeof(double));
    knn.nidx = (int*) malloc(knnSize * sizeof(int));

    const int xSize = m * d;
    const int ySize = n * d;

    //Init distances
    int distancesSize = n * m;
    double* D = (double *) malloc(distancesSize * sizeof(double));

    //calculate sum(X.^2, 2) and put it to A
    double* A = (double *) malloc(m * sizeof(double));

    for(int i=0;i<m;i++){
        //Calculate sqrt(sum(X.^2, 2)) faster for large d values
        A[i] = cblas_dnrm2(d, &X[i * d], 1);
        //Get the power of 2
        A[i] = A[i] * A[i];
    }

    //calculate sum(Y.^2, 2).' and put it to C
    //C is transposed
    double* C = (double *) malloc(n * sizeof(double));

    for(int j=0;j<n;j++){
        //Calculate sqrt(sum(Y.^2, 2)) faster for large d values
        C[j] = cblas_dnrm2(d, &Y[j * d], 1);
        //Get the power of 2
        C[j] = C[j] * C[j];
    }

    //calculate  (- 2 * X*Y.') and put it to B
    double* B = (double *) malloc(m * n * sizeof(double));

    // void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
    //              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
    //              const int K, const double alpha, const double *A,
    //              const int lda, const double *B, const int ldb,
    //              const double beta, double *C, const int ldc);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2, X, d, Y, d, 0, B, n);

    //Calculate D
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            D[i * n + j] = sqrt(fabs(A[i] + B[i * n + j] + C[j]));
        }
    }

    free(A);
    free(B);
    free(C);

    //Init yids O(n * m)
    int* yId = (int*) malloc(distancesSize * sizeof(int));

    for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			yId[i * n + j] = j;
		}
    }

    //Calculate knn
    for(int i=0;i<m;i++){
		//Quicksort D array and move id elements the same way
        quickSort(D, yId, i * n, i * n + n - 1, k);

        //Keep only the k shortest distances
        for(int j=0;j<k;j++){
            knn.ndist[i * k + j] = D[i * n + j];
            knn.nidx[i * k + j] = yId[i * n + j];
        }
    }

    free(yId);
    free(D);

    return knn;
}