//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Vecotr adder                                                *
//Description: This program is for testing CPU performance with vector *
//             add function.                                           *
//                                                                     *
//                                                                     *
//File Name: pb1_cpu.c                                                 *
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


void random_ints(int* r, int n);

int main()
{
	int n=10000000;
	int i;
 	int *a, *b, *c;
    //int n = 1024 * 1024;
    int size = n * sizeof(int);
    struct timeval start, end;
    float timer;
    gettimeofday(&start, NULL);
    
	//srand((unsigned)time(NULL));
    //Alloc space for host copies of a, b, c and setup input values  
    a = (int *)malloc(size); random_ints(a, n);
    b = (int *)malloc(size); random_ints(b, n);
    c = (int *)malloc(size); 
    
    for(i=0;i<n;i++)
    {
        c[i] = a[i] + b[i];
   // printf("%d\n",c[i]);
    }
    gettimeofday(&end, NULL);
    timer = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Number of loop is: %d\nRunning time is: %2f ms\n", n,timer/1000);
    free(a);
    free(b);
    free(c);
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
