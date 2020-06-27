//**********************************************************************
// Function Name: convolution (Kernel)                                 *
// Description: - Execute direct(naive) convolution_local memory       *
//              - OPENCL                                               *
// Input file: none                                                    *
// Output file: none                                                   *
// Return: none                                                        *
//**********************************************************************

__kernel void convolute(__global unsigned char* output,
	__global unsigned char* image,
	__global float* flt, int imageGemmRgbSize, int filterSize, int channel)
{
	int col;//global id 0
	int i, j;
	int r, g, b; //multiply and sum of filter and data

	col = get_global_id(0);

	__local float filter[7776]; // upto 9*9*96 (filerSize* channel size)
	if (i < filterSize*filterSize*channel)
	{
		filter[i]=flt[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (col < imageGemmRgbSize*channel)
	{
		r = 0;
		g = 0;
		b = 0;

		for (i = 0; i < channel; i++)
		{
			for (j = 0; j < filterSize * filterSize; j++)
			{
				r += filter[i*channel + j] * image[col*filterSize*filterSize*3];		//R
				g += filter[i*channel + j] * image[col*filterSize*filterSize*3 + 1];	//G
				b += filter[i*channel + j] * image[col*filterSize*filterSize*3 + 2];	//B
			}
		output[col * i * 3] = r;
		output[col * i * 3 + 1] = g;
		output[col * i * 3 + 2] = b;
		}
	}
}