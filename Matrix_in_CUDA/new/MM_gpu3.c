//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing GPU performance with multi- *
//             dimention multiple function_tiled.                      *
//                                                                     *
//                                                                     *
//File Name: MM_gpu3.cu                                                *
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
#define TILE_SIZE 16

// kernel on GPU for  matrixMul tiled

__global__ void matrixMul_tiled(int *A_gpu,int *B_gpu, int *C_gpu, int A_rows, int A_columns, int B_columns)
{
	__shared__ int shared_A_tile[TILE_SIZE][TILE_SIZE];
	__shared__ int shared_B_tile[TILE_SIZE][TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i,j,sum;

	if (row<A_rows && col<B_columns)
	{

		for (i = 0; i <= A_columns/TILE_SIZE; i++)
		{
			// Tile both matrix A and B
			shared_A_tile[ty][tx] = A_gpu[row * A_columns + i* TILE_SIZE + tx];
			shared_B_tile[ty][tx] = B_gpu[(i * TILE_SIZE + ty) * B_columns + col];
			__syncthreads();
			
			for (j = 0; j < TILE_SIZE; j++)
			{
				if (j + (i * TILE_SIZE) < A_columns) 
				{
					sum += (shared_A_tile[ty][j] * shared_B_tile[j][tx]);
				}
			}
		}	
		C_gpu[row * B_columns + col] = sum;
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

    matrixMul_tiled<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,M,N,K); //C matrix: K(width) x M(height)

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
            r[i * b + j] = rand() % 10;
        }
    }

}
