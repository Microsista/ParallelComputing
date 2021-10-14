#include <device_launch_parameters.h>

__global__ void sum(unsigned char* input, unsigned char* output, int workSize, int width);