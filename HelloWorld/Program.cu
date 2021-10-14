#include "Globals.cuh"
#include "Utilities.cuh"
#include "Sum.cuh"

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
	

		void* inputOnDevice;
		void* outputOnDevice;

		auto inputSizeInBytes = width * height * sizeof(Uchar3);
		auto numberOfGroups = (inputSizeInBytes + numberOfThreads - 1) / numberOfThreads;

		Uchar3* output = new Uchar3[inputSizeInBytes];

		auto allocateInputAndOutputOnDevice = [&] {
			ThrowIfFailed(cudaMalloc(&inputOnDevice, inputSizeInBytes));
			ThrowIfFailed(cudaMalloc(&outputOnDevice, inputSizeInBytes));
		};

		auto copyInputToDevice = [&] {
			ThrowIfFailed(cudaMemcpy(inputOnDevice, img, inputSizeInBytes, cudaMemcpyHostToDevice));
		};

		auto executeSumKernel = [&] {
			sum<<<numberOfGroups, numberOfThreads>>>(reinterpret_cast<unsigned char*>(inputOnDevice), reinterpret_cast<unsigned char*>(outputOnDevice), inputSizeInBytes, width);
			ThrowIfFailed(cudaGetLastError());
		};

		auto copyAndShowOutputFromDevice = [&] {
			ThrowIfFailed(cudaMemcpy(output, outputOnDevice, inputSizeInBytes, cudaMemcpyDeviceToHost));
		};

		auto freeBuffersOnDevice = [&] {
			ThrowIfFailed(cudaFree(inputOnDevice));
			ThrowIfFailed(cudaFree(outputOnDevice));
		};

		allocateInputAndOutputOnDevice();
		copyInputToDevice();
		executeSumKernel();
		copyAndShowOutputFromDevice();
		stbi_write_jpg("output.jpg", width, height, 3, output, 100);

		freeBuffersOnDevice();
		delete output;
	}
	catch (exception& e){
		cerr << e.what();
	}
}