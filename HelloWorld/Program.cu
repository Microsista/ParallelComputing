#include "Utilities.cuh"
#include "Sum.cuh"
#include "SeparateChannels.cuh"
#include "RecombineChannels.cuh"

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
#include <chrono>

using namespace std;


int main() {
	try {
		int blurRadius = 7;
		int numCols, numRows, channels;
		unsigned char* img = stbi_load("assets\\ptak.jpg", &numCols, &numRows, &channels, 3);
		if (img == NULL) {
			printf("error loading the image\n");
			exit(1);
		}
		printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", numCols, numRows, channels);


		uchar3* d_inputImageRGB;
		uchar3* d_outputImageRGB;

		unsigned char* d_red;
		unsigned char* d_green;
		unsigned char* d_blue;

		unsigned char* d_redBlurred;
		unsigned char* d_greenBlurred;
		unsigned char* d_blueBlurred;

		auto inputSizeInBytes = numCols * numRows * sizeof(Uchar3);

		Uchar3* output = new Uchar3[inputSizeInBytes];

		auto allocateInputAndOutputOnDevice = [&] {
			ThrowIfFailed(cudaMalloc(&d_inputImageRGB, inputSizeInBytes));
			ThrowIfFailed(cudaMalloc(&d_outputImageRGB, inputSizeInBytes));

			ThrowIfFailed(cudaMalloc(&d_red, inputSizeInBytes / 3));
			ThrowIfFailed(cudaMalloc(&d_green, inputSizeInBytes / 3));
			ThrowIfFailed(cudaMalloc(&d_blue, inputSizeInBytes / 3));

			ThrowIfFailed(cudaMalloc(&d_redBlurred, inputSizeInBytes / 3));
			ThrowIfFailed(cudaMalloc(&d_greenBlurred, inputSizeInBytes / 3));
			ThrowIfFailed(cudaMalloc(&d_blueBlurred, inputSizeInBytes / 3));
		};

		auto copyInputToDevice = [&] {
			ThrowIfFailed(cudaMemcpy(d_inputImageRGB, img, inputSizeInBytes, cudaMemcpyHostToDevice));
		};

		auto copyOutputFromDevice = [&] {
			ThrowIfFailed(cudaMemcpy(output, d_outputImageRGB, inputSizeInBytes, cudaMemcpyDeviceToHost));
		};

		auto freeBuffersOnDevice = [&] {
			ThrowIfFailed(cudaFree(d_inputImageRGB));
			ThrowIfFailed(cudaFree(d_outputImageRGB));

			ThrowIfFailed(cudaFree(d_red));
			ThrowIfFailed(cudaFree(d_green));
			ThrowIfFailed(cudaFree(d_blue));

			ThrowIfFailed(cudaFree(d_redBlurred));
			ThrowIfFailed(cudaFree(d_greenBlurred));
			ThrowIfFailed(cudaFree(d_blueBlurred));
		};

		allocateInputAndOutputOnDevice();
		copyInputToDevice();

		const dim3 blockSize(16, 16, 1);
		const dim3 gridSize{ numCols / blockSize.x + 1, numRows / blockSize.y + 1, 1 };

		auto t1 = chrono::high_resolution_clock::now();
		separateChannels<<<gridSize, blockSize>>>(d_inputImageRGB, numRows, numCols, d_red, d_green, d_blue);
		cudaDeviceSynchronize(); ThrowIfFailed(cudaGetLastError());

		
		sum<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, blurRadius);
		sum<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, blurRadius);
		sum<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, blurRadius);
		cudaDeviceSynchronize(); ThrowIfFailed(cudaGetLastError());

		recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGB, numRows, numCols);
		cudaDeviceSynchronize(); ThrowIfFailed(cudaGetLastError());
		auto t2 = chrono::high_resolution_clock::now();

		cout << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "\n";

		copyOutputFromDevice();
		stbi_write_jpg("output.jpg", numCols, numRows, 3, output, 100);

		freeBuffersOnDevice();
		delete output;
	}
	catch (exception& e){
		cerr << e.what();
	}
}