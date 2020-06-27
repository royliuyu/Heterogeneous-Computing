//**********************************************************************
// Function Name: convolution (Kernel)                                 *
// Description: - Execute direct(naive) convolution_global memory      *
//              - OPENCL                                               *
// Input file: none                                                    *
// Output file: none                                                   *
// Return: none                                                        *
//**********************************************************************

__kernel void convolute(__global unsigned char* output,
	__global unsigned char* image,
	__global float* flt, int imageGemmRgbSize, int filterSize)
{
	int col;//global id 0
	int i, j;
	int r, g, b; //multiply and sum of filter and data

	col = get_global_id(0);

	__local float filter[81];
	if (i < filterSize*filterSize)
	{
		filter[i] = flt[i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (col < imageGemmRgbSize)
	{
		r = 0;
		g = 0;
		b = 0;

		for (i = 0; i < filterSize * filterSize; i++)
		{
				r += filter[i] * image[col*3];		//R
				g += filter[i] * image[col*3 + 1];	//G
				b += filter[i] * image[col*3 + 2];	//B
		}
		output[col*3] = r;
		output[col*3 + 1] = g;
		output[col*3 + 2] = b;
	}
}