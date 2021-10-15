#include <device_launch_parameters.h>

__global__ void recombineChannels(const unsigned char* d_redBlurred, const unsigned char* d_greenBlurred, const unsigned char* d_blueBlurred, uchar3* d_outputImageRGB, int numRows, int numCols);