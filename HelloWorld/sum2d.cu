#include "Globals.cuh"

#include <device_launch_parameters.h>

__global__ void sum2d(int* input, int* output, int workSize) {
	const auto sharedWidth = numberOfThreads + 2 * blurRadius;
	__shared__ int shared[sharedWidth * sharedWidth];
	auto globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
	auto localIndex = threadIdx.x + blurRadius;

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