#include "Globals.cuh"

#include <device_launch_parameters.h>

#include <algorithm>

using namespace std;

__global__ void sum(unsigned char* input, unsigned char* output, int workSize, int width) {
	auto& indexInGroup = threadIdx.x;
	__shared__ unsigned char shared[numberOfThreads + 2 * blurRadius];
	int globalIndex = indexInGroup + blockIdx.x * blockDim.x;
	int localIndex = indexInGroup + blurRadius;
	
	if (globalIndex < workSize) {
		auto loadToShared = [&] {
			auto actual = [&] {
				shared[localIndex] = input[globalIndex + indexInGroup * 3];
			};

			auto halo = [&] {
				if (indexInGroup < blurRadius) {
					shared[indexInGroup] = input[max(0, globalIndex - blurRadius)];
					shared[localIndex + numberOfThreads] = input[min(workSize, globalIndex + numberOfThreads)];
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