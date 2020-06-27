//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing GPU performance with vector *
//             add function.                                           *
//                                                                     *
//                                                                     *
//File Name: pb1c_gpu.c                                                 *
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
#include<cuda_runtime.h>  
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

__global__ void add(int a[], int b[], int c[])
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void random_ints(int* r, int n);
int main()
{
	//for counting run time
    struct timeval start, end;
    float timer;
    gettimeofday(&start, NULL);
    
    int*a, *d_a, *b, *d_b, *c, *d_c;
   	int n=10000;
    
    int size = n * sizeof(int);

    // data initializing
    a = (int *)malloc(size); random_ints(a, n);
    b = (int *)malloc(size); random_ints(b, n);
    c = (int *)malloc(size); 
    //for (int i=0;i<n;i++)	printf("%d\n",a[i]);//for testing
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // CPU TO GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
 
   // Define kernel,block:(1024*1024/512)£¬512 threds each block

    dim3 dimGrid(n/512);
    dim3 dimBlock(512); //each block has X threads

    // kernel
    add<<<dimGrid, dimBlock>>>(d_a,d_b,d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    gettimeofday(&end, NULL);
    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Number of loop is: %d\nRunning time is: %f ms\n", n,timer/1000);

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
 	for (i=0; i < n; ++i)
 	{
 	 	r[i] = rand()/2;
 	 	
 	}

}
