//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Convolution                                                 *
//Description: This program is to do convolution calculation           *
//             - OpenCL                                                *
//             - Direct (naive) convolution                            *
//                                                                     *
//File Name: naivecon.c , naiveconv_kernel.cl                          *
//File Version: 1.0                                                    *
//Baseline: Homework_2                                                 *
//                                                                     *
//Course: ECGR 6090 Heterogeneous Computing                            *
//                                                                     *
//Programmed by: Yu Liu                                                * 
//Under Suppervision of: Dr. Hamed Tabkhi                              *
//                                                                     *
//Input file: images/viptraffic0.ppm ...  images/viptraffic119.ppm     *
//Output file: none                                                    *
//**********************************************************************  

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>
#include <time.h>

#define HEIGHT 160
#define WIDTH 120
#define FLTSIZE 3 //filter size
#define PADDING 0
#define STRIDE 1


//**********************************************************************
// Function Name: decode_image                                         *
// Description: - read image in ppm formate, read the data of array    *
//                named frame[]                                        *
// Input file: image file : viptrafficX.ppm                            *
// Output file: none                                                   *
// Return: 0 if success                                                *
//**********************************************************************

int decode_image(unsigned char frame[HEIGHT * WIDTH * 3], char filename[])
{
	FILE *pFile;
	pFile = fopen(filename, "r");
	fseek(pFile, 15L, SEEK_SET);//In ppm file, the first 15 bytes are content of "p6,120 160, 255", image data is from 16th bytes

	fread(frame, sizeof(unsigned char), HEIGHT * WIDTH * 3+15, pFile); 
	fclose(pFile);
	return 0;
}

//**********************************************************************
// Function Name:LoadOpenCLKernel                                      *
// Description: - read kernel file, get the code and input to array    *
//                named buf                                            *
// Input file: image file : viptrafficX.ppm                            *
// Output file: none                                                   *
// Return: kernel file size                                            *
//**********************************************************************
long LoadOpenCLKernel(char const* path, char **buf)
{
	FILE  *fp;
	size_t fsz;
	long   off_end;
	int    rc;

	/* Open the file */
	fp = fopen(path, "r");
	if (NULL == fp) {
		return -1L;
	}

	/* Seek to the end of the file */
	rc = fseek(fp, 0L, SEEK_END);
	if (0 != rc) {
		return -1L;
	}

	/* Byte offset to the end of the file (size) */
	if (0 > (off_end = ftell(fp))) {
		return -1L;
	}
	fsz = (size_t)off_end;

	/* Allocate a buffer to hold the whole file */
	*buf = (char *)malloc(fsz + 1);
	if (NULL == *buf) {
		return -1L;
	}

	/* Rewind file pointer to start of file */
	rewind(fp);

	/* Slurp file into buffer */
	if (fsz != fread(*buf, 1, fsz, fp)) {
		free(*buf);
		return -1L;
	}

	/* Close the file */
	if (EOF == fclose(fp)) {
		free(*buf);
		return -1L;
	}

	/* Make sure the buffer is NUL-terminated, just in case */
	(*buf)[fsz] = '\0';

	//printf("%s\n", *buf); //for debugging
	//for debugging , check kernel code be transfered
	//FILE *outputFile;//Open file to outpur result
	//char *filename = "kernelcode.txt";
	//outputFile = fopen(filename, "w");
	//fprintf(outputFile, "%s", *buf);
	//fprintf(outputFile, "\n");
	//fflush(outputFile);
	//fclose(outputFile);

	/* Return the file size */
	return (long)fsz;
}

//**********************************************************************
// Function Name:randomInit                                            *
// Description: - Generate random value to an float array              *
//                                                                     *
// Input file: none                                                    *
// Output file: none                                                   *
// Return: kernel file size                                            *
//**********************************************************************
int randomInit(float* data, int size, int range) // random form 0/255 to 255/255
{
	int i;
	srand(time(NULL));
	for (i = 0; i < size; i++) 
	{
		data[i] = rand() % range / (float)range;
	}
	//for (i = 0; i < size; i++) printf("%f;", data[i]); // for debugging
	return 0;
}

//**********************************************************************
// Function Name:Main                                                  *
// Description: - Main function on host, configure the kernel parameter*
//                and run kernel                                       *
// Input file: none                                                    *
// Output file: none                                                   *
// Return: 0 if success                                                *
//**********************************************************************
int main(int argc, char** argv) {

	//define memory size of host and kernel
	
	int convWidth = (WIDTH - FLTSIZE + 2 * PADDING) / STRIDE + 1;  //convolution width with padding
	int convHeight = (HEIGHT - FLTSIZE + 2 * PADDING) / STRIDE + 1;  //convolution width with padding
	int imagecount = 0; //counter for 120 images	
	int err;

	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                   // compute kernel

	cl_mem d_image; //input image
	cl_mem d_filter; //filter
	cl_mem d_output; //output image

	//set timer
	cl_event myevent; //timing - profiling
	cl_ulong start = 0; //event start
	cl_ulong end = 0; //event stop
	cl_float kernelExecTimeNs = 0; //measure time

	float* filter = (float*)malloc(FLTSIZE*FLTSIZE * sizeof(float));
	unsigned char* image = (unsigned char*)malloc(HEIGHT * WIDTH * sizeof(unsigned char) * 3);
	unsigned char* output= (unsigned char*)malloc(convHeight * convWidth * 3 * sizeof(unsigned char));
	randomInit(filter, FLTSIZE*FLTSIZE, 255); //initialize filter
	//printf("%f,\n", filter);


	printf("Initializing OpenCL device...\n");

	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);

	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

	// Connect to a compute device
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source file
	char *KernelSource;
	long lFileSize;
	//lFileSize = LoadOpenCLKernel("naiveconv_globalMM_kernel.cl", &KernelSource); //for naiveconvolution global memory
	lFileSize = LoadOpenCLKernel("naiveconv_globalMM_kernel.cl", &KernelSource); //for naiveconvolution local memory

	//printf("%s\n", KernelSource);//for debugging

	if (lFileSize < 0L) {
		perror("File read failed");
		return 1;
	}

	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable! Error code: %d\n",err);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "convolute", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	//interation to calculate with all images
	while (imagecount<120) {
		
		char filename[50];//file length upto 50
		sprintf(filename, "images/viptraffic%d.ppm", imagecount);//read viptrafficX.ppm
		decode_image(image, filename); //get image data from file
		imagecount++;

		////for debugging , check kernel code be transfered
		//	FILE *outputFile;//Open file to outpur result
		//	char *filename = "kernelcode.txt";
		//	outputFile = fopen(filename, "w");
		//	int i;
		//	for (i = 0; i < 10; i++)
		//	{
		//		fprintf(outputFile, "%c", image[i]);
		//	}
		//	fprintf(outputFile, "\n");
		//	fflush(outputFile);
		//	fclose(outputFile);

		//Create buffer for device, put RGB together
		d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, convHeight * convWidth * 3 * sizeof(unsigned char), NULL, &err);
		d_image = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT*WIDTH * sizeof(unsigned char)*3, image, &err);
		d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FLTSIZE*FLTSIZE * sizeof(int), filter, &err);

		if (!d_image || !d_filter || !d_output)
		{
			printf("Error: Failed to allocate device memory!\n");
			exit(1);
		}

		err = clEnqueueWriteBuffer(commands, d_image, CL_TRUE, 0, HEIGHT*WIDTH * sizeof(unsigned char) * 3, image, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FLTSIZE*FLTSIZE * sizeof(int), filter, 0, NULL, NULL);

		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to write data to device! %d\n", err);
			exit(1);
		}

		int imageWidth = WIDTH;
		int imageHeight = HEIGHT;
		int filterSize = FLTSIZE;
		int padding = PADDING;
		int stride = STRIDE;
		size_t localWorkSize[2], globalWorkSize[2];
		err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
		err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&imageWidth);
		err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&imageHeight);
		err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filterSize);
		err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&padding);
		err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&stride);

		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to set kernel arguments! %d\n", err);
			exit(1);
		}

		localWorkSize[0] = 16;
		localWorkSize[1] = 16;
		globalWorkSize[0] = 12*32; //image width is 120*3
		globalWorkSize[1] = 4*32; //image heitht is 160

		err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to execute kernel! %d\n", err);
			exit(1);
		}
		clWaitForEvents(1, &myevent);
		clFinish(commands);
		clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		kernelExecTimeNs += (end - start);
		
		//Retrieve result from device
		err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, convWidth * convHeight * 3 * sizeof(unsigned char), output, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to read output array! %d\n", err);
			exit(1);
		}
	}
	printf("Cumputing done!  Programed by OpenCL.\n");
	printf("Image amount:%d;  Image size:%d x %d;  Padding:%d;  Stride:%d;  Filter Size:%d.\n", imagecount, WIDTH, HEIGHT, PADDING, STRIDE, FLTSIZE);
	printf("Kernel Execution time: %f milli seconds\n", kernelExecTimeNs / 1000000);

	//Shutdown and cleanup
	free(image);
	free(filter);
	free(output);

	clReleaseMemObject(d_image);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_filter);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
