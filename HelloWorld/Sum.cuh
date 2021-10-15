#include <device_launch_parameters.h>

__global__ void sum(unsigned char* d_color, unsigned char* d_colorBlurred, int numRows, int numCols, int blurRadius);