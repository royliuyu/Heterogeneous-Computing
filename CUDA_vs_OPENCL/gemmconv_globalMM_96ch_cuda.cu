//**********************************************************************
//                                                                     *
//               University Of North Carolina Charlotte                *
//                                                                     *
//Program: Convolution                                                 *
//Description: This program is to do convolution calculation           *
//             - CUDA                                                  *
//             - GEMM convolution , global memory                      *
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
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCKSIZE 256
#define HEIGHT 160
#define WIDTH 120
#define FLTSIZE 7 //filter size
#define PADDING 0
#define STRIDE 2
#define CHANNEL 96

//**********************************************************************
// Function Name: convolution (Kernel)                                 *
// Description: - Execute direct(naive) convolution                    *
//              - CUDA_global memory                                   *
// Input file: none                                                    *
// Output file: none                                                   *
// Return: none                                                        *
//**********************************************************************
__global__ void convolution(unsigned char *image_d, unsigned char *output_d, float* filter, int imageGemmRgbSize, int filterSize, int channel)
{
	int i, j, col;
	int r, g, b;

	col = blockIdx.x * blockDim.x + threadIdx.x; //image width *3

	if (col < imageGemmRgbSize*channel)
	{
		r = 0;
		g = 0;
		b = 0;

		for (i = 0; i < channel; i++)
		{
			for (j = 0; j < filterSize * filterSize; j++)
			{
				r += filter[i*channel + j] * image_d[col * filterSize * filterSize * 3];		//R
				g += filter[i*channel + j] * image_d[col * filterSize * filterSize * 3 + 1];	//G
				b += filter[i*channel + j] * image_d[col * filterSize * filterSize * 3 + 2];	//B
			}
		output_d[col * i * 3] = r;
		output_d[col * i * 3 + 1] = g;
		output_d[col * i * 3 + 2] = b;
		}
	}
}

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

	fread(frame, sizeof(unsigned char), HEIGHT * WIDTH * 3 + 15, pFile);
	fclose(pFile);
	return 0;
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
// Function Name:  transpose_gemm                                      *
// Description: - transpose image to GEMM                              *
//                RGB chnannel                                         *
// Input file: none                                                    *
// Output file: none                                                   *
// Return: 0 if success                                                *
//**********************************************************************
int transpose_gemm_rgb(unsigned char* input, unsigned char* output)
{
	int i, j, k, step;
	int convline = 0;
	for (i = 0; i < (HEIGHT - FLTSIZE + 1); i+=STRIDE) // Height iteration
	{
		for (j = 0; j < (WIDTH - FLTSIZE + 1) * 3; j += (3*STRIDE)) //Width iteration
		{
			for (k = 0; k < FLTSIZE*FLTSIZE ; k ++)
			{
				step = (i*STRIDE + k / FLTSIZE)*WIDTH * 3 + (j*STRIDE + k%FLTSIZE);
				//output_2D[convline][k]=input[step]; //2D, 1 channel
				output[convline] = input[step]; //R
				output[convline + 1] = input[step + 1]; //G
				output[convline + 2] = input[step + 2]; //B
				convline += 3;
			}
		}
	}
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
int main(void)
{
	int filterSize = FLTSIZE;
	int channel = CHANNEL;
	int convWidth = (WIDTH - FLTSIZE + 2 * PADDING) / STRIDE + 1;  //convolution width with padding
	int convHeight = (HEIGHT - FLTSIZE + 2 * PADDING) / STRIDE + 1;  //convolution heigth with padding
	int imageRgbSize = HEIGHT * WIDTH * 3;  // value is 57600 when image 160*120
	int imageGemmRgbSize = convWidth * convHeight * FLTSIZE * FLTSIZE * 3;  //value is 645183 when image 160*120, filter =7, padding =0, stride =2
	int outputSize = convHeight * FLTSIZE * FLTSIZE * CHANNEL * 3; //value is 1086624 when image 160*120, filter =7, padding =0, stride =2, channel =3
	int imagecount = 0; //counter for 120 images	
	unsigned char *image_d, *output_d;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float kernelExecTime = 0;
	float timer;

	float* filter = (float*)malloc(FLTSIZE*FLTSIZE * CHANNEL * sizeof(float));
	unsigned char* image = (unsigned char*)malloc(imageRgbSize * sizeof(unsigned char));
	unsigned char* imageGemmRgb = (unsigned char*)malloc(imageGemmRgbSize * sizeof(unsigned char));
	unsigned char* output = (unsigned char*)malloc(outputSize * sizeof(unsigned char));
	randomInit(filter, FLTSIZE*FLTSIZE*CHANNEL, 255); //initialize filter

	////for debugging
	//int k;
	//for (k = 0; k < FLTSIZE*FLTSIZE*CHANNEL; k++)
	//{
	//	printf("filter[%d]: %f;    ", k, filter[k]);
	//}


	cudaMalloc((void**)&image_d, imageGemmRgbSize * sizeof(unsigned char));
	cudaMalloc((void**)&output_d, outputSize * sizeof(unsigned char));

	while (imagecount < 120) 
	{
		char filename[50];//file length upto 50
		sprintf(filename, "images/viptraffic%d.ppm", imagecount);//read viptrafficX.ppm
		decode_image(image, filename); //get image data from file
		transpose_gemm_rgb(image, imageGemmRgb); 
		imagecount++;

		////for debugging
		//int k;
		//for (k = 65000; k < 65100; k++)  //value is 645183 when image 160*120, filter =7, stride =2
		//{
		//	printf("image[%d]: %d;    ", k, imageGemmRgb[k]);
		//}

		//Copy from host to device
		cudaMemcpy(image_d, imageGemmRgb, imageGemmRgbSize, cudaMemcpyHostToDevice);

		dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
		dim3 dimGrid((imageGemmRgbSize + BLOCKSIZE - 1) / BLOCKSIZE,(imageGemmRgbSize + BLOCKSIZE - 1) / BLOCKSIZE);

		cudaEventRecord(start, 0);
		convolution <<<dimGrid, dimBlock >>> (image_d, output_d, filter, imageGemmRgbSize, filterSize, channel);
	
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		//Copy from device to host
		cudaMemcpy(output, output_d, outputSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		
		////for debugging
		//int k;
		//for (k = 10000; k < 10010; k++)
		//{
		//	printf("output[%d]: %f;    ", k, output[k]);
		//}

		cudaEventElapsedTime(&timer, start, stop);
		kernelExecTime += timer;
	}

	//Free memory allocation
	cudaFree(output_d);
	cudaFree(image_d);
	free(output);
	free(image);
	free(imageGemmRgb);
	free(filter);
	

	printf("Cumputing done!  Golbal memory applied in CUDA.\n");
	printf("Image amount:%d;  Image size:%d x %d;  Padding:%d;  Stride:%d;  Filter Size:%d.\n", imagecount, WIDTH, HEIGHT, PADDING, STRIDE, FLTSIZE);
	printf("Kernel Execution time: %f milli seconds\n", kernelExecTime);
	//system("pause");

	return EXIT_SUCCESS;
}
