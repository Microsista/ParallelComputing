#include "Globals.cuh"
#include "Utilities.cuh"
#include "Sum.cuh"
#include "Add.cuh"
#include "Sum2d.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;


int main() {
	try {
		int width, height, channels;
		unsigned char* img = stbi_load("assets\\ptak.jpg", &width, &height, &channels, 3);
		if (img == NULL) {
			printf("error loading the image\n");
			exit(1);
		}
		printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

		struct int3 {
			int x, y, z;
			int3(int x, int y, int z) : x{ x }, y{ y }, z{ z } {}
		};

		std::vector<int> input;
		for (auto i = 0; i < width; i++) {
			for (auto j = 0; j < height; j++) {
				const stbi_uc* p = img + (3 * (j * width + i));
				input.push_back(p[0]);
				input.push_back(p[1]);
				input.push_back(p[2]);
			}
		}

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
			ThrowIfFailed(cudaGetLastError());
		};

		auto copyAndShowOutputFromDevice = [&] {
			ThrowIfFailed(cudaMemcpy(output.data(), outputOnDevice, inputSizeInBytes, cudaMemcpyDeviceToHost));
		};

		auto freeBuffersOnDevice = [&] {
			ThrowIfFailed(cudaFree(inputOnDevice));
			ThrowIfFailed(cudaFree(outputOnDevice));
		};

		allocateInputAndOutputOnDevice();
		copyInputToDevice();
		executeSumKernel();
		copyAndShowOutputFromDevice();
		stbi_write_jpg("output.jpg", width, height, 3, output.data(), 100);

		freeBuffersOnDevice();
	}
	catch (exception& e){
		cerr << e.what();
	}
}