//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing GPU performance with vector *
//             add function.                                           *
//                                                                     *
//                                                                     *
//File Name: pb1b_gpu.c                                                 *
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
	int n=1000000;
	//for counting run time
    struct timeval start, end;
    float timer;
    gettimeofday(&start, NULL);
    
    int *d_a, *d_b, *d_c;

    //int n = 1024 * 1024;
    int size = n * sizeof(int);

    // data initializing
    d_a = (int *)malloc(size); random_ints(d_a, n);
    d_b = (int *)malloc(size); random_ints(d_b, n);
    d_c = (int *)malloc(size); 
    
	dim3 dimGrid(n/512);
    dim3 dimBlock(512); //each block has X threads   

    // kernel
    add<<<dimGrid, dimBlock>>>(d_a,d_b,d_c);

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
