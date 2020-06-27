//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing CPU performance with vector *
//             add function.                                           *
//                                                                     *
//                                                                     *
//File Name: pb2_cpu.c                                                 *
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

#define N          10000000
#define RADIUS     2


void random_ints(int* r, int n);

int main()
{
	//for counting run time
    struct timeval start, end;
    float timer;
    gettimeofday(&start, NULL);
    
	int*in, *out;
   	int n,i,j;
	n = N;
   
    int size = (n+2*RADIUS)*sizeof(int);

    // data initializing
    in = (int *)malloc(size); random_ints(in, n);
    out = (int *)malloc(size); 
    
    
	// for (i=0;i<(n+1+2*RADIUS);i++)  	out[i]=0;  //for testing
	

	
	for (i=0;i<n;i++)
	{
		for (j=0; j<(1+2*RADIUS);j++)
		{
			out[i] += in[i+j];
		}
	}
	
	// cleanup
	free(in);
	free(out);
	
 	gettimeofday(&end, NULL);
    timer = 10000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Data number is: %d\nRadius is: %d\nRunning time is: %f ms\n", n,RADIUS,timer/1000);
    
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
