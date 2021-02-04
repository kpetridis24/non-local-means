
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
struct timespec t0, t1;

double*  addNoise(double* Im, int imSize, double scalar);
double   GaussianNoise(double sigma, double x);
double** createMatrix(int row, int col);
double** readFile(char* filename, int rowSize, int colSize);
void     writeFile(double**A, char* filename, int rowSize, int colSize);
double** oneDim2twoDim(double* A, int len);
double*  twoDim2oneDim(double** A, int len);
void     printArray(double **A, double *B, int len, int dim);
double*  GaussianKernel(int krnl_sz, double sigma);
void     findPatch(double* Im, int imSize, int row, int col, int ptSize, double* patch);
double** NonLocalMeans(double* Im, int imSize, int ptSize, double imSgm, double ptSgm);


void main(int argc, char** argv){

    int    patch_sz    = 5,   
           image_sz    = 64,
           size = pow(image_sz, 2) * pow(patch_sz, 2);
    double patch_sgm   = 0.8, 
           sigma       = 0.08,
           noiseEffect = 0.1;
    
    double** Im    = readFile("house.txt", image_sz, image_sz);
    double*  Im2   = twoDim2oneDim(Im, image_sz);
    double*  noise = addNoise(Im2, image_sz*image_sz, noiseEffect);
    double** If    = NonLocalMeans(noise, image_sz, patch_sz, sigma, patch_sgm);
    writeFile(If, "denoise.txt", image_sz, image_sz);
}


/* Returns patch of the pixel(row,col) */
void findPatch(double* Im, int imSize, int row, int col, int ptSize, double* patch){

    int r, r2, c, c2, cnt = 0, range = (ptSize - 1) / 2;

    for(r=row-range, r2=0; r2 < ptSize; r++, r2++)
        for(c=col-range, c2=0; c2 < ptSize; c++, c2++)
        { 
            if((r >= imSize) || (c >= imSize)) patch[cnt++] = -1;
            else if(r < 0  || c < 0) patch[cnt++] = -1;
            else if(r >= 0 || c >= 0) patch[cnt++] = Im[r*imSize+c];
        }
}


/* Image denoising */
double** NonLocalMeans(double* Im, int imSize, int ptSize, double imSgm, double ptSgm){

    double x, x2, tmp, D=0, normZ=0;
    int size = pow( imSize, 2 ),
        len  = pow( ptSize, 2 );
    
    double*  ptWeight = GaussianKernel(ptSize, ptSgm);
    double*  W        = (double *)malloc(size * sizeof(double));
    double*  If       = (double *)malloc(size * sizeof(double));
    double*  patch1   = (double *)malloc(len  * sizeof(double));  
    double*  patch2   = (double *)malloc(len  * sizeof(double));

    //clock_gettime(CLOCK_REALTIME, &t0);
    for(int e = 0; e < imSize; e++){   
        for(int e2 = 0; e2 < imSize; e2++)
        {
        
            findPatch(Im, imSize, e, e2, ptSize, patch1);

            for(int i = 0; i < imSize; i++){   
                for(int i2 = 0; i2 < imSize; i2++)
                {   

                    findPatch(Im, imSize, i, i2, ptSize, patch2);
                    
                    for(int p = 0; p < ptSize; p++)
                        for(int p2 = 0; p2 < ptSize; p2++)
                        {
                            x  = patch1[p*ptSize+p2];
                            x2 = patch2[p*ptSize+p2];
                            if(x != -1 && x2 != -1)
                            {  
                                tmp = pow( (x-x2), 2 );       
                                tmp *= ptWeight[p*ptSize+p2];  
                                D   += tmp;  
                            }
                        }
            
                    D  = exp( (-D) / pow(imSgm, 2) );
                    W[i*imSize+i2] = D;
                    normZ += D;
                    D = 0;
                }
            }
        
            for(int v = 0; v < size; v++)
            {
                W[v] /= normZ;
                If[e*imSize+e2] += W[v] * Im[v];
            }
            normZ = 0;
        }
    }
    //clock_gettime(CLOCK_REALTIME, &t1);
    //double duration = ((t1.tv_sec-t0.tv_sec)*1000000+(t1.tv_nsec-t0.tv_nsec)/1000)/1000000.0;
    //printf("~ Duration: %f sec\n", duration);

    double** Ifilt = oneDim2twoDim(If, imSize);
    return Ifilt;
}


/* Calculates spacial-gaussian weight */
double* GaussianKernel(int krnl_sz, double sigma){

    double *W = (double *)malloc(pow( krnl_sz, 2 ) * sizeof(double)),
             x, y, d, sum = 0.0,
             c = 2 * pow( sigma, 2 );

    for(int i = 0; i < krnl_sz; i++)
        for(int j = 0; j < krnl_sz; j++)
        {   
            x = i - (krnl_sz - 1) / 2.0;
            y = j - (krnl_sz - 1) / 2.0;
            d = x * x + y * y;
            W[i*krnl_sz+j] = exp( -(d) / c ) / (M_PI * c);
            sum += W[i*krnl_sz+j];
        }

    double max[krnl_sz];
    int i = 0;

    for(i = 0, max[i] = 0; i < krnl_sz; i++)
        for(int j = 0; j < krnl_sz; j++)
        {
            W[i*krnl_sz+j] /= sum;
            if(j==0) max[i] = W[i*krnl_sz+j];
            else if(W[i*krnl_sz+j] > max[i]) 
                max[i] = W[i*krnl_sz+j]; 
        }

    for(int i = 0; i < krnl_sz; i++)
        for(int j = 0; j < krnl_sz; j++)
            W[i*krnl_sz+j] /= max[i];

    return W;
}


/* Prints 1-D / 2-D */
void printArray(double **A, double *B, int len, int dim){

    if(dim == 2){
        for(int i = 0; i < len; i++){
            for(int j = 0; j < len; j++){
                printf("%f ", A[i][j]);
            }
            printf("\n");
        }
    }
    else{
        for(int i = 0; i < len; i++) printf("%f\n", B[i]);
    }
}


/* 2D to 1D */
double* twoDim2oneDim(double** A, int len){

    double* C = (double *)malloc(len * len * sizeof(double ));
    int cnt = 0;

    for(int i = 0; i < len; i++)
        for(int j = 0; j < len; j++)
            C[cnt++] = A[i][j];

    return C;
}


/* 1D to 2D */
double** oneDim2twoDim(double* A, int len){

    double** D = (double **)malloc(len * sizeof(double*));
    for(int f = 0; f < len; f++) D[f] = malloc(len * sizeof(double));
    int cnt = 0;

    for(int i = 0; i < len; i++)
        for(int j = 0; j < len; j++)
            D[i][j] = A[cnt++];

    return D;
}


double** readFile(char* filename, int rowSize, int colSize){

    double** A = (double **)malloc(rowSize * sizeof(double));
    for(int i=0; i<colSize; i++) A[i] = (double *)malloc(colSize * sizeof(double));

    FILE *fp = fopen(filename, "r");

    for(int i = 0; i < rowSize; i++)
        for(int j = 0; j < colSize; j++)
        {
            fscanf(fp, "%lf %*c", &A[i][j]);
        }

    return A;
}


void writeFile(double**A, char* filename, int rowSize, int colSize){

    FILE *fp = fopen(filename, "w");

    for(int i = 0; i < rowSize; i++){
        for(int j = 0; j < colSize; j++)
        {
            fprintf(fp, "%lf,", A[i][j]);
        }
        fprintf(fp, "\n");
    }
}


/* 2-D random matrix */
double** createMatrix(int row, int col){

    //srand(time( NULL ));
    double** mat = (double **)malloc(row * sizeof(double));
    for(int i=0; i<row; i++) mat[i] = malloc(col * sizeof(double));

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            mat[i][j] = (double)( rand() ) / (double)( RAND_MAX );

    return mat;
}


double* addNoise(double* Im, int imSize, double scalar){

    double *noise = (double *)malloc(imSize * sizeof(double)),
            value, effect;

    for(int i = 0; i < imSize; i++)
    {
        value    = ((double)( rand() ) / RAND_MAX*20 - 10);
        effect   = GaussianNoise(2, value) - 0.1;
        noise[i] = (scalar*effect + 1) * Im[i];
    }
    
    return noise;
}


double GaussianNoise(double sigma, double x){

    return (1 / (sigma*sqrt(2*M_PI)))*exp((-x*x) / (2*sigma*sigma));
}

