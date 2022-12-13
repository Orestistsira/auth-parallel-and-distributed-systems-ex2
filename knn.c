#include "knn.h"

#define print(num) printf("%d\n", num)

void printArrayDouble(double* arr, int size){
    for(int i=0;i<size;i++){
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

void printArrayInt(int* arr, int size){
    for(int i=0;i<size;i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int* copyArray(int const* src, int len)
{
   int* p = malloc(len * sizeof(int));
   memcpy(p, src, len * sizeof(int));
   return p;
}

void swapDouble(double *a, double *b) {
  double t = *a;
  *a = *b;
  *b = t;
}

void swapInt(int *a, int *b) {
  int t = *a;
  *a = *b;
  *b = t;
}

int partition(double* array, int* otherArray, int low, int high) {
  
  // select the rightmost element as pivot
  int pivot = array[high];
  
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

void quickSort(double* array, int* otherArray, int low, int high) {
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

knnresult kNN(double* X, double* Y, int n, int m, int d, int k){
    //Init knn
    knnresult knn;
    knn.k = k;
    knn.m = m;

    int knnSize = k * m;
    knn.ndist = (double*) malloc(knnSize * sizeof(double));
    knn.nidx = (int*) malloc(knnSize * sizeof(int));

    const int xSize = m * d;
    const int ySize = n * d;

    //Init yids O(n)
    int* yId = (int*) malloc(n * sizeof(int));

    for(int j=0;j<n;j++){
        yId[j] = j;
    }

    printf("yId: ");
    printArrayInt(yId, n);

    //Init distances O(m x n x d)
    //TODO: calculate distances with other equation
    int distancesSize = n * m;
    double* distances = (double *) malloc(distancesSize * sizeof(double));
    
    int xIndex = 0;
    for(int x=0;x<xSize;x+=d){
        int yIndex = 0;
        for(int y=0;y<ySize;y+=d){
            double sum = 0;

            for(int i=0;i<d;i++){
                sum += pow(X[x + i] - Y[y + i], 2);
            }

            distances[xIndex * m + yIndex] = sqrt(sum);

            yIndex++;
        }

        xIndex++;
    }

    printf("distances: ");
    printArrayDouble(distances, distancesSize);

    //calculate knn
    double* xDist = (double*) malloc(n * sizeof(double));

    for(int i=0;i<m;i++){

        for(int j=0;j<n;j++){
            xDist[j] = distances[i * n + j];
        }

        printArrayDouble(xDist, n);
        //TODO: do not copy array
        int* yIdTemp = copyArray(yId, n);

        quickSort(xDist, yIdTemp, 0, n - 1);

        printArrayDouble(xDist, n);
        printArrayInt(yIdTemp, n);

        
        for(int j=0;j<k;j++){
            knn.ndist[i * k + j] = xDist[j];
            knn.nidx[i * k + j] = yIdTemp[j];
        }

             
    }
    free(xDist);

    printf("result:\n");
    printArrayDouble(knn.ndist, m * k);
    printArrayInt(knn.nidx, m * k);  


    free(yId);
    free(distances);

    return knn;
}