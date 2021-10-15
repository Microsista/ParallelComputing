#include <device_launch_parameters.h>


__global__ void recombineChannels(const unsigned char* d_redBlurred, const unsigned char* d_greenBlurred, const unsigned char* d_blueBlurred,
	uchar3* d_outputImageRGB, int numRows, int numCols) {

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red = d_redBlurred[thread_1D_pos];
	unsigned char green = d_greenBlurred[thread_1D_pos];
	unsigned char blue = d_blueBlurred[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar3 outputPixel = make_uchar3(red, green, blue);

	d_outputImageRGB[thread_1D_pos] = outputPixel;
}