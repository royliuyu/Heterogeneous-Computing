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
	__global float* flt, int imageWidth, int imageHeight, int filterSize, int padding, int stride)
{
	int col;//global id 0
	int row;//global id 1
	int i, j;
	int r, g, b; //multiply and sum of filter and data
	int convWidth = (imageWidth - filterSize + 2 * padding) / stride + 1;

	col = get_global_id(0);
	row = get_global_id(1);
	r = 0;
	g = 0;
	b = 0;

	__local float filter[81];
	if ( i < filterSize & j< filterSize)
		{
			filter[i*filterSize + j] = flt[i*filterSize + j];
		}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (col < (imageWidth - padding + 1)*3 && row <(imageHeight - padding + 1))
	{
		for (i = 0; i < filterSize; i++)
		{
			for (j = 0; j < filterSize; j++)
			{
				r += filter[i*filterSize + j] * image[3 * (row + i) * imageWidth + col + j]; //R
				g += filter[i*filterSize + j] * image[3 * (row + i) * imageWidth + col + j + 1];//G
				b += filter[i*filterSize + j] * image[3 * (row + i) * imageWidth + col + j + 2]; //B
			}
		}
		output[row * convWidth + col] = r;
		output[row * convWidth + col + 1] = g;
		output[row * convWidth + col + 2] = b;
	}
}