//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing GPU performance with one    *
//             stencil.                                                *
//                                                                     *
//                                                                     *
//File Name: pb2b_gpu.cu                                                 *
//File Version: 1.0                                                    *
//Baseline: Homework_0                                                 *
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
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<cuda_runtime.h>  

#define N          10000
#define RADIUS     8
#define BLOCK_SIZE 128

void random_ints(int *r, int n);

__global__ void stencil_1d(int *in, int *out)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	
	int j;
		int result=0;
		for (j=0; j<(1+2*RADIUS);j++)
		{
			result += in[i];
			in+=j;
		}
		
 	out[i]=result;
}

int main()
{
	//for counting run time
    struct timeval start, end;
    float timer;
    gettimeofday(&start, NULL);
    
	int*in, *d_in, *out, *d_out;
   	int n;
	n = N;
   
    int size = (n+2*RADIUS)*sizeof(int);

    // data initializing
    in = (int *)malloc(size); random_ints(in, n);
    out = (int *)malloc(size); 


    //for (int i=0;i<n;i++)	printf("%d\n",a[i]);//for testing
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // CPU TO GPU
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
   
   // Define kernel,block:(1024*1024/512)£¬512 threds each block
    dim3 dimGrid(n/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE); //each block has X threads

    // kernel
    stencil_1d<<<dimGrid, dimBlock>>>(d_in, d_out);
	
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	// cleanup
	free(in);
	free(out);
	cudaFree(d_in);
	cudaFree(d_out);
	
 	gettimeofday(&end, NULL);
    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Data number is: %d\nBlocksize is: %d\nRadius is: %d\nRunning time is: %f ms\n", n,BLOCK_SIZE,RADIUS,timer/1000);
     
    return 0;
}

//**********************************************************************
// Function Name: random_ints                                          *
// Description: - Generate random integer                              *
// Input : None                                                        *
// Output : Random integer                                             *
// Return: None                                                        *
//**********************************************************************
void random_ints(int* r, int n)
{
	int i;
 	for (i=0; i < n+2*RADIUS; ++i)
 	{
 	 	r[i] = rand()/2;
 	 	
 	}

}
