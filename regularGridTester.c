#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv){
    int d, n, k;
    if(argc == 4){
        n = atoi(argv[1]);
        d = atoi(argv[2]);
        k = atoi(argv[3]);
    }
    else{
        printf("Error: give n d k as arguments\n");
        exit(-1);
    }

    int id;
    double dist;
    FILE* f;

    f = fopen("knn_result.txt", "r");

    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            fscanf(f, "%d %lf", &id, &dist);

            if(j == 0){
                if(dist != 0){
                    printf("Wrong result\n");
                    exit(0);
                }
            }
            else if(j <= d){
                if(dist != 1){
                    printf("Wrong result\n");
                    exit(0);
                }
            }
        }
    }

    printf("Correct result\n");

    fclose(f);
}