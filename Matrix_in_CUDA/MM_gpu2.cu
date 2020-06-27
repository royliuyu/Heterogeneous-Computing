//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing GPU performance with multi- *
//             dimention multiple function_coaleced.                   *
//                                                                     *
//                                                                     *
//File Name: MM_gpu2.cu                                                *
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
            A: M x N   M(height), N(width),
            B: N x K   N(height), K(width),
            C: M X K   M(height), K(width), 
            

*/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define M 10000
#define N 10000
#define K 10000
#define BLOCK_SIZE 16


// kernel on GPU device, coalesced

__global__ void matrixMul_coalesced(int *A_gpu,int *B_gpu, int *C_gpu, int A_rows, int A_columns, int B_columns)
{
  __shared__ int shared_A_coal[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ int shared_B_coal[BLOCK_SIZE*BLOCK_SIZE];

  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int width = gridDim.x * BLOCK_SIZE;

  for (int i = 0; i< BLOCK_SIZE; i += BLOCK_SIZE) //Blocksize X= Block size Y
	{
    shared_A_coal[threadIdx.x*BLOCK_SIZE + threadIdx.y + i] = A_gpu[(row+i)*width + col];
    shared_B_coal[threadIdx.x*BLOCK_SIZE + threadIdx.y + i] = B_gpu[(col+i)*width + row];
  	}

  __syncthreads();

  col = blockIdx.y * BLOCK_SIZE + threadIdx.x;
  row = blockIdx.x * BLOCK_SIZE + threadIdx.y;

  for (int j = 0; j < BLOCK_SIZE; j += BLOCK_SIZE) 
  	{
    C_gpu[(row+j)*width + col] = shared_A_coal[threadIdx.x + (threadIdx.y+j)*BLOCK_SIZE]*shared_B_coal[threadIdx.x + (threadIdx.y+j)*BLOCK_SIZE];
  	}
}

void random_ints(int* r, int a, int b);
int main()
{
    int *A_cpu, *B_cpu, *C_cpu;
    int *A_gpu, *B_gpu, *C_gpu;
        
	//for counting run time
    struct timeval start, end;
    float timer;
    gettimeofday(&start, NULL);
    
   // memcopy A_cpu to A_gpu
    //memcopy B_cpu to B_gpu
    A_cpu =(int*)malloc(sizeof(int)*M*N);
	random_ints(A_cpu,M,N);
    B_cpu =(int*)malloc(sizeof(int)*K*N);
	random_ints(B_cpu,N,K);
    C_cpu =(int*)malloc(sizeof(int)*M*K);
    
    cudaMalloc((void**)&A_gpu, sizeof(int)*M*N);
    cudaMalloc((void**)&B_gpu, sizeof(int)*N*K);
    cudaMalloc((void**)&C_gpu, sizeof(int)*M*K);
    
    cudaMemcpy(A_gpu, A_cpu, sizeof(int)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_cpu, sizeof(int)*N*K, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMul_coalesced<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,M,N,K); //C matrix: K(width) x M(height)

    //memcopy C_gpu to C_cpu
    cudaMemcpy(C_cpu, C_gpu, sizeof(int)*M*K, cudaMemcpyDeviceToHost);
     
     
    free(A_cpu);
    free(B_cpu);
    free(C_cpu);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    
    cudaDeviceSynchronize(); 
    gettimeofday(&end, NULL);
    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Number of loop is: %dx%dx%d.\nRunning time is: %f ms\n", M,N,K,timer/1000);

    return 0;
}

void random_ints(int* r, int a, int b)
{	srand(time(0));
	int i,j;
    for (i = 0; i < a; ++i) {
        for (j = 0; j < b; ++j) {
            r[i * b + j] = rand() % 100;
        }
    }

}
