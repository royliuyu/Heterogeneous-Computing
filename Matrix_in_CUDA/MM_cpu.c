//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing CPU performance with multi- *
//             dimention multiple function.                            *
//                                                                     *
//                                                                     *
//File Name: pb1_cpu.c                                                 *
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
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#define M 10000
#define N 10000
#define K 10000

int main() 
{
	int i,j,k,r,tmp;
	int A[100];
	int B[100];
	int C[100];
	
	struct timeval start, end;
    float timer;
    gettimeofday(&start, NULL);

	//optimized Matrix multiplication in c
	for(r=0; r<M; r++) 
		{
    	for (i=0; i<N; i++) 
			{
			for (j=0; j<N; j++) 
				{
				tmp = A[i*N+j];

				for (k=0; k<N; k++) 
					{
					C[i*N+K] += tmp * B[j*N+K];
					}
    			}
			}	
    	}

	/* without optimization
	int m, k, n;
	int A[M][N], B[N][K], C[M][K];
	
    struct timeval start, end;
    float timer;
    gettimeofday(&start, NULL);
    
	for(m=0;m<M;m++)
    {
		for(k=0; k<K; k++)
		{
			for (n=0; n<N; n++)
			{
				C[m][k]= C[m][k]+A[m][n]*B[n][k];
			}
						
		}
    }
    */
    gettimeofday(&end, NULL);
    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Loop size is %d*%d*%d.\nRunning time is: %2f ms\n", M,N,K,timer/1000);

    return 0;
}


