
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

struct timeval tstart;
struct timeval tic();
double toc(struct timeval begin);
double*  addNoise(double* Im, int imSize, double scalar);
double   GaussianNoise(double sigma, double x);
double** createMatrix(int row, int col);
double** readFile(char* filename, int rowSize, int colSize);
void     mywriteFile(double**A, char* filename, int rowSize, int colSize);
double** oneDim2twoDim(double* A, int len);
double*  twoDim2oneDim(double** A, int len);
void     printArray(double **A, double *B, int len, int dim);
double*  GaussianKernel(int krnl_sz, double sigma);


//NUM_THREADS SHOULD BE IMAGE_SIZE^2
__global__ void NonLocalMeans(double* Im, int imSize, double imSigma, double* patch, 
                                          int ptSize, double* ptW, double* W, double* If){
    
    double x, x2, tmp, D = 0, normZ=0;
    int i2 = 0, j3 = 0,
        size = pow( imSize, 2 ),
        len  = pow( ptSize, 2 ),
        id   = threadIdx.x + blockDim.x * blockIdx.x;

    for(int i = 0; i < size; i++)
    {
        for(int j=i*len, j2=0; j < i*len+len; j++, j2++)
        {  
            if(j3 == ptSize){
                i2 ++;
                j3 = 0;
            }
            if(i2 == ptSize) i2 = 0;
    
            x  = patch[id*len+j2];                   
            x2 = patch[j];
            if(x != 0 && x2 != 0)
            {   
                tmp = pow( (x-x2), 2 );         
                tmp *= ptW[i2*ptSize+j3];   
                D   += tmp;  
            }
            j3++;          
        }

        D = exp( (-D) / pow(imSigma, 2) );
        W[id] = D;
        If[id] += W[id] * Im[i];
        normZ += D;
        D = 0;
    }

    If[id] /= normZ; 
   
}


__global__ void findPatches(double* Im, double* patch, int imageSize,
                                        int patchSize){
    
    int size = pow( imageSize, 2 ) * pow( patchSize, 2 );
    int r, r2, c, c2, cnt = 0, range = (patchSize - 1) / 2;

    for(int i = 0; i < imageSize; i++)
        for(int j = 0; j < imageSize; j++)
        {   
            for(r=i-range, r2=0; r2 < patchSize; r++, r2++)
                for(c=j-range, c2=0; c2 < patchSize; c++, c2++)
                { 
                    if((r >= imageSize) || (c >= imageSize)) patch[cnt++] = 0;
                    else if(r < 0  || c < 0) patch[cnt++] = 0;
                    else if(r >= 0 || c >= 0) patch[cnt++] = Im[r*imageSize+c];
                }
        }
}



int main(){
    
    int    pSize   = 7,
           iSize   = 128,
           size    = pow(iSize, 2),
           nblocks = size / 512;
    double pSigma  = 0.8,
           iSigma  = 0.08,
           nEffect = 0.5;
    
    double** nIm  = readFile("im128.txt", iSize, iSize);

    double* Im    = twoDim2oneDim(nIm, iSize);
    double* noise = addNoise(Im, size, nEffect);
    double* krnl  = GaussianKernel(pSize, pSigma);
    double* If    = (double *)malloc(size * sizeof(double));

    double *dnoise, *dkrnl, *patch, *W, *dfilt;
    int size1 = pow( iSize, 2 ) * sizeof(double),
        size2 = pow( pSize, 2 ) * sizeof(double),
        size3 = pow( iSize, 2 ) * pow( pSize, 2 ) * sizeof(double);

    cudaMalloc((void **)&dnoise, size1);
    cudaMalloc((void **)&dfilt , size1);
    cudaMalloc((void **)&dkrnl , size2);
    cudaMalloc((void **)&patch , size3);
    cudaMalloc((void **)&W     , size1);
    
    cudaMemcpy(dnoise, noise, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(dkrnl, krnl, size2, cudaMemcpyHostToDevice);
    
    tstart = tic();

    findPatches<<<1,1>>>(dnoise, patch, iSize, pSize);
    NonLocalMeans<<<nblocks,512>>>(dnoise, iSize, iSigma, patch, pSize, dkrnl, W, dfilt);
    cudaDeviceSynchronize();
    
    double duration = toc(tstart);
    printf("~ Duration: %f sec\n", duration);

    cudaMemcpy(If, dfilt, size1, cudaMemcpyDeviceToHost);

    double** If2 = oneDim2twoDim(If, iSize);
    mywriteFile(If2, "denoise.txt", iSize, iSize);

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



void printArray(double **A, double *B, int len, int dim){

    if(dim == 2)
    {
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


void mywriteFile(double**A, char* filename, int rowSize, int colSize){

    FILE *fp = fopen(filename, "w");

    for(int i = 0; i < rowSize; i++){
        for(int j = 0; j < colSize; j++)
        {
            fprintf(fp, "%lf,", A[i][j]);
        }
        fprintf(fp, "\n");
    }
}


double GaussianNoise(double sigma, double x){

    return (1 / (sigma*sqrt(2*M_PI)))*exp((-x*x) / (2*sigma*sigma));
}


double* addNoise(double* Im, int imSize, double scalar){

    double *noise = (double *)malloc(imSize * sizeof(double)),
            value, effect;

    for(int i = 0; i < imSize; i++)
    {
        value    = ((double)( rand() ) / RAND_MAX*20 - 10);
        effect   = GaussianNoise(2, value) - 0.1;
        noise[i] = (scalar * effect + 1) * Im[i];
    }
    
    return noise;
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
    for(int f = 0; f < len; f++) D[f] = (double *)malloc(len * sizeof(double));
    int cnt = 0;

    for(int i = 0; i < len; i++)
        for(int j = 0; j < len; j++)
            D[i][j] = A[cnt++];

    return D;
}



/* 2-D random matrix */
double** createMatrix(int row, int col){

    srand(time( NULL ));
    double** mat = (double **)malloc(row * sizeof(double *));
    for(int i=0; i<row; i++) mat[i] = (double *)malloc(col * sizeof(double));

    for(int i=0; i<row; i++)
        for(int j=0; j<col; j++)
            mat[i][j] = (double)( rand() ) / (double)( RAND_MAX );

    return mat;
}


struct timeval tic(){
    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv;
}

double toc(struct timeval begin){
    
    struct timeval end;
    gettimeofday(&end, NULL);
    double stime = ((double)(end.tv_sec-begin.tv_sec)*1000)+
                            ((double)(end.tv_usec-begin.tv_usec)/1000);
    stime /= 1000;
    return stime;
}


