#include "Globals.cuh"

#include <device_launch_parameters.h>

#include <algorithm>

using namespace std;

__global__ void sum(unsigned char* input, unsigned char* output, int workSize, int width) {
	__shared__ unsigned char shared[numberOfThreads + 2 * blurRadius];
	int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int localIndex = threadIdx.x + blurRadius;
	
	if (globalIndex < workSize) {
		auto loadToShared = [&] {
			auto actual = [&] {
				shared[localIndex] = input[globalIndex + threadIdx.x * 3];
			};

			auto halo = [&] {
				if (threadIdx.x < blurRadius) {
					/*shared[localIndex - blurRadius] = globalIndex + threadIdx.x * 3 >= blurRadius * 3 ? input[globalIndex + threadIdx.x * 3 - blurRadius * 3] : 0;
					shared[localIndex + numberOfThreads] = globalIndex + threadIdx.x * 3 < (workSize - numberOfThreads * 3) ? input[globalIndex * threadIdx.x * 3 + numberOfThreads * 3] : 0;*/
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
		output[globalIndex] = sumNeighbouringValues() / (2 * blurRadius + 1);
	}
}