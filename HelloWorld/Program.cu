#include "Globals.cuh"
#include "Utilities.cuh"
#include "Sum.cuh"
#include "Add.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

using namespace std;

int main() {
	try {
		vector<int> input{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		vector<int> output(input.size());

		void* inputOnDevice;
		void* outputOnDevice;

		auto inputSizeInBytes = input.size() * sizeof(input[0]);
		auto numberOfGroups = (input.size() + numberOfThreads - 1) / numberOfThreads;

		auto allocateInputAndOutputOnDevice = [&] {
			ThrowIfFailed(cudaMalloc(&inputOnDevice, inputSizeInBytes));
			ThrowIfFailed(cudaMalloc(&outputOnDevice, inputSizeInBytes));
		};

		auto copyInputToDevice = [&] {
			ThrowIfFailed(cudaMemcpy(inputOnDevice, input.data(), inputSizeInBytes, cudaMemcpyHostToDevice));
		};

		auto executeSumKernel = [&] {
			sum<<<numberOfGroups, numberOfThreads>>>(reinterpret_cast<int*>(inputOnDevice), reinterpret_cast<int*>(outputOnDevice), input.size());
		};

		auto executeAddKernel = [&] {
			add<<<numberOfGroups, numberOfThreads>>>(reinterpret_cast<int*>(inputOnDevice), reinterpret_cast<int*>(inputOnDevice),
				reinterpret_cast<int*>(outputOnDevice), input.size());
		};

		auto copyAndShowOutputFromDevice = [&] {
			ThrowIfFailed(cudaMemcpy(output.data(), outputOnDevice, inputSizeInBytes, cudaMemcpyDeviceToHost));
			print(output);
		};

		auto freeBuffersOnDevice = [&] {
			ThrowIfFailed(cudaFree(inputOnDevice));
			ThrowIfFailed(cudaFree(outputOnDevice));
		};

		ThrowIfFailed(cudaSetDevice(1));
		allocateInputAndOutputOnDevice();
		copyInputToDevice();
		executeSumKernel();
		copyAndShowOutputFromDevice();
		freeBuffersOnDevice();
	}
	catch (exception& e){
		cerr << e.what();
	}
}