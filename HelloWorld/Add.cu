#include <device_launch_parameters.h>

__global__ void add(int* a, int* b, int* c, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
		c[i] = a[i] + b[i];
}