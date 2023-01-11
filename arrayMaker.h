#include <stdio.h>
#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <string.h>

double* getArrayFromTxt(char* filename, int numOfpoints, int dimensions, int startingRow, int endingRow){
    FILE* f;

    int pointsToRead = endingRow - startingRow;
    double* Y = (double*) malloc(pointsToRead * dimensions * sizeof(double));

    if((f = fopen(filename, "r")) == NULL){
        printf("File: %s not found\n", filename);
        exit(1);
    }

    int x = 0;
    double unusedVar;
    // Read the data of matrix
    for (int i=0;i<numOfpoints;i++){
        if(i < startingRow){
            for (int j=0;j<dimensions;j++){
                if(fscanf(f, "%lf", &unusedVar) <= 0){
                    printf("Error scanning file\n");
                    exit(1);
                }
            }
            continue;
        } 
        if(i >= endingRow) break;

        for (int j=0;j<dimensions;j++){
            if(fscanf(f, "%lf", &Y[x * dimensions + j]) <= 0){
                // ERROR("Error scanning file");
                // exit(1);
                //to avoid nan values and leave room for sum(DBL_MAX * DBL_MAX) from 0 to d
                Y[x * dimensions + j] = sqrt(DBL_MAX) / dimensions;
            }
        }
        x++;
    }

    fclose(f);

    return Y;
}

double* getRandomArray(int numOfPoints, int dimensions){
    int size = numOfPoints * dimensions;
    double* Y = (double*) malloc(size * sizeof(double));

    for(int i=0;i<size;i++){
        double val = ((double) (rand() % 100));
        Y[i] = val;
    }

    return Y;
}

double* getMinstArray(char* filepath, int startingRow, int endingRow){
    int fileRows;
    int d = 784;
    int n = endingRow - startingRow;
    int infoLength = 4;

    char substring[6];
    strcpy(substring, "train");

    if(strstr(filepath, substring) != NULL){
        fileRows = 60000;
    }
    else{
        fileRows = 10000;
    }
    //fileRows = 10;

    int i, j, k, fd;
    unsigned char *ptr;

    if ((fd = open(filepath, O_RDONLY)) == -1) {
        fprintf(stderr, "couldn't open image file");
        exit(-1);
    }

    int imageInfo[infoLength];
    
    if(read(fd, imageInfo, infoLength * sizeof(int)) <= 0){
        printf("Error in reading file\n");
    }

    //free(imageInfo);

    unsigned char charData[n][d];
    
    // read-in mnist numbers (pixels|labels)
    unsigned char unusedBuffer[d];
    int x = 0;
    for(i=0;i<fileRows;i++) {
        if(i < startingRow){
            if(read(fd, unusedBuffer, d * sizeof(unsigned char)) <= 0){
                printf("Error in reading file\n");
            };
            continue;
        } 
        if(i >= endingRow) break;

        if(read(fd, charData[x], d * sizeof(unsigned char)) <= 0){
            //printf("Error in reading file\n");
            for(int j=0;j<d;j++){
                charData[x][j] = (char)(sqrt(DBL_MAX) / d * 255);
            }
        }
        x++;
    }

    close(fd);

    double* Y = (double*) malloc(n * d * sizeof(double));

    for(i=0;i<n;i++){
        for(j=0;j<d;j++){
            Y[i * d + j] = (double)charData[i][j] / 255.0;
        }
    }

    return Y;
}

void printResult(knnresult* knn, int n, const char* filename){
    int k = knn->k;
    FILE* f;

    f = fopen(filename, "w");

    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            fprintf(f, "%d %f ", knn->nidx[i * k + j], knn->ndist[i * k + j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}