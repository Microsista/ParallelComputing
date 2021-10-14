#include "Globals.cuh"

#include <device_launch_parameters.h>

__global__ void sum(int* input, int* output, int workSize) {
	__shared__ int shared[numberOfThreads + 2 * blurRadius];
	int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int localIndex = threadIdx.x + blurRadius;
	
	if (globalIndex < workSize) {
		auto loadToShared = [&] {
			auto actual = [&] {
				shared[localIndex] = input[globalIndex];
			};

			auto halo = [&] {
				if (threadIdx.x < blurRadius) {
					shared[localIndex - blurRadius] = globalIndex >= blurRadius ? input[globalIndex - blurRadius] : 0;
					shared[localIndex + numberOfThreads] = globalIndex < (workSize - numberOfThreads) ? input[globalIndex + numberOfThreads] : 0;
				}
			};

			actual();
			halo();
			__syncthreads(); 
		};

		auto sumNeighbouringValues = [&] {
			int result = 0;
			for (int offset = -blurRadius; offset <= blurRadius; offset++) {
				result += shared[localIndex + offset];
			}
			return result;
		};
		
		loadToShared();
		output[globalIndex] = sumNeighbouringValues();
	}
}