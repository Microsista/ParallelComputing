#ifndef UTILITIES_H
#define UTILITIES_H

#include "CudaException.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#define ThrowIfFailed(cudaStatus) { \
	if (cudaStatus != cudaSuccess) \
		throw CudaException(cudaStatus, __FILE__, __LINE__); \
}

void print(const std::vector<int>& vector) {
	for (auto& element : vector)
		std::cout << element << " ";
	std::cout << "\n";
}

void print(const int* array, int size) {
	for (auto i = 0; i < size; i++)
		std::cout << array[i] << " ";
	std::cout << "\n";
}

//struct int3 {
//	int x, y, z;
//	int3(int x, int y, int z) : x{ x }, y{ y }, z{ z } {}
//};

#endif