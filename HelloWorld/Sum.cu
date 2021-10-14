#include "Globals.cuh"
#include "Utilities.cuh"
#include <device_launch_parameters.h>

#include <algorithm>

using namespace std;

__global__ void sum(Uchar3* input, Uchar3* output, int workSize, int width) {
	struct Uchar3 {
		unsigned char x, y, z;
		Uchar3() {}
		Uchar3(unsigned char x, unsigned char y, unsigned char z) : x{ x }, y{ y }, z{ z } {}
		Uchar3 operator*(const Uchar3& rhs) {
			return Uchar3(this->x * rhs.x, this->y * rhs.y, this->z * rhs.z);
		}
		Uchar3 operator*(float rhs) {
			return Uchar3(this->x * rhs, this->y * rhs, this->z * rhs);
		}
		Uchar3 operator+(const Uchar3& rhs) {
			return Uchar3(this->x + rhs.x, this->y + rhs.y, this->z + rhs.z);
		}
	};

	int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int globalX = globalIndex % width;
	int globalY = globalIndex / width;

	Uchar3 result;
	/*for (int offset = -blurRadius; offset <= blurRadius; offset++) {
		result += input[min(workSize, max(0, globalIndex + offset))];
	}*/

	float weights[] = {
		0.05f, 0.09f, 0.12f, 0.15f, 0.16f, 0.15f, 0.12f, 0.09f, 0.05f
	};
	for (int i = -blurRadius; i <= blurRadius; i++) {
		for (int j = -blurRadius; j <= blurRadius; j++) {
			int x = j, y = i;
			if (globalIndex % width + j < 0)
				x = 0;
			if (globalIndex % width + j > width)
				x = 0;
			if (globalIndex / width + i < 0)
				y = 0;
			if (globalIndex / width + i > workSize/width)
				y = 0;

			result = result + *input[globalX + x + (width * (globalY + y))] * weights[j + blurRadius] * weights[i + blurRadius];
		}
	}

	*output[globalIndex] = result;
}