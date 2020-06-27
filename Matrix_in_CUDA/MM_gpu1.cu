//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing GPU performance with multi- *
//             dimention multiple function_naive.                      *
//                                                                     *
//                                                                     *
//File Name: MM_gpu1.cu                                                *
//File Version: 1.0                                                    *
//Baseline: Homework_1                                                 *
//                                                                     *
//Course: ECGR6090- Heterogeneous Computing                            *
//                                                                     *
//Programmed by: Roy Liu                                               * 
//Under Suppervision of: Dr. Hamed Tabkhi                              *
//                                                                     *
//Input file: No                                                       *
//                                                                     *
//Output:Time of program running                                       *
//**********************************************************************  

/*
Matrix:
            A: m x n
            B: n x k 
            C: m X k 
*/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
//#include<cuda_runtime.h> 

#define M 100
#define N 100
#define K 100

#define BLOCK_SIZE 16

// kernel on GPU device

__global__ void matrixMul_naive(int *A_gpu,int *B_gpu, int *C_gpu,int A_rows, int A_columns, int B_columns)
{
	
 	int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum,i;
    if( col < B_columns && row < A_rows) //scope of matrix C
    {
	    for(i = 0; i < A_columns; i++) 
	    {
	        sum = sum + A_gpu[row * A_columns + i] * B_gpu[i * B_columns + col];
	    }
	    C_gpu[row * B_columns + col] = sum;
    }
} 
void random_ints(int* r, int a, int b);
int main()
{

	
    int *A_cpu, *B_cpu, *C_cpu;
    int *A_gpu, *B_gpu, *C_gpu;
    
   // memcopy A_cpu to A_gpu
    //memcopy B_cpu to B_gpu
    
    A_cpu =(int*)malloc(sizeof(int)*M*N);
	random_ints(A_cpu,M,N);
    B_cpu =(int*)malloc(sizeof(int)*K*N);
	random_ints(B_cpu,N,K);
    C_cpu =(int*)malloc(sizeof(int)*M*K);
	
	//for debugging
	//for (i=0;i<m;i++) printf("%d,",A_cpu[i]);
    //printf("\n");
	//for (i=0;i<m*k;i++) printf("%d,",C_cpu[i]);
	
   	//count time
	struct timeval start, end;
    float timer;
 	//float cuda_time;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);	
	//cudaEventRecord(start,0);
	
	gettimeofday(&start, NULL);
  	
    cudaMalloc((void**)&A_gpu, sizeof(int)*M*N); 
    cudaMalloc((void**)&B_gpu, sizeof(int)*N*K);
    cudaMalloc((void**)&C_gpu, sizeof(int)*M*K);
    cudaMemcpy(A_gpu, A_cpu, sizeof(int)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_cpu, sizeof(int)*N*K, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    


	//printf("before a&b: %d\n %d \n",A_cpu,B_cpu);
	//printf("before c: %d \n",C_cpu);
    matrixMul_naive<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,M,N,K);
	
    //memcopy C_gpu to C_cpu
    cudaMemcpy(C_cpu, C_gpu, sizeof(int)*M*K, cudaMemcpyDeviceToHost);
    
    //for debugging
	//printf("after a&b: %d\n %d \n",A_cpu,B_cpu);
	//for (i=0;i<m;i++) printf("%d,",B_cpu[i]);
 	//printf("\n");
	//printf("after c: %d \n",C_cpu);
	//for (i=0;i<m*k+2;i++) printf("%d,",C_cpu[i]);
	
	
	cudaDeviceSynchronize(); 
	gettimeofday(&end, NULL);
    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Loop size is %d*%d*%d.\nRunning time is: %2f ms\n", M,N,K,timer/1000);
    
	//cudaEventRecord(stop,0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&cuda_time, start, stop);
	//printf("GPU's elapsed time:%f ms.\n", cuda_time);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

 	
    free(A_cpu);
    free(B_cpu);
    free(C_cpu);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

 	printf("Data size: %d x %d x %d.\n",M,N,K);

	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

    return 0;
}

void random_ints(int* r, int a, int b)
{	srand(time(0));
	int i,j;
    for (i = 0; i < a; ++i) {
        for (j = 0; j < b; ++j) {
            r[i * b + j] = rand() % 10;
        }
    }

}
