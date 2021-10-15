#include <device_launch_parameters.h>

__global__ void separateChannels(const uchar3* inputImageRGB, int numRows, int numCols,
	unsigned char* const redChannel, unsigned char* greenChannel, unsigned char* blueChannel) {

	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= numCols || py >= numRows) {
		return;
	}
	int i = py * numCols + px;
	redChannel[i] = inputImageRGB[i].x;
	greenChannel[i] = inputImageRGB[i].y;
	blueChannel[i] = inputImageRGB[i].z;
}