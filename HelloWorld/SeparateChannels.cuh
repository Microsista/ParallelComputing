#include <device_launch_parameters.h>

__global__ void separateChannels(const uchar3* inputImageRGB, int numRows, int numCols,
	unsigned char* const redChannel, unsigned char* greenChannel, unsigned char* blueChannel);