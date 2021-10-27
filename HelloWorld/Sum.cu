#include <device_launch_parameters.h>

using namespace std;

__global__ void sum(unsigned char* d_color, unsigned char* d_colorBlurred, int numRows, int numCols, int blurRadius) {
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float weights[] = {
		0.05f, 0.09f, 0.12f, 0.15f, 0.16f, 0.15f, 0.12f, 0.09f, 0.05f
	};
	int color = 0;
	for (int i = -blurRadius; i <= blurRadius; i++) {
		for (int j = -blurRadius; j <= blurRadius; j++) {
			int x = j, y = i;
			if (thread_2D_pos.x + j <= 0)
				x = 0;
			if (thread_2D_pos.x + j > numCols)
				x = 0;
			if (thread_2D_pos.y + i <= 0)
				y = 0;
			if (thread_2D_pos.y + i > numRows)
				y = 0;

			int2 sample2DPos = make_int2(blockIdx.x * blockDim.x + threadIdx.x + x,
				blockIdx.y * blockDim.y + threadIdx.y + y);

			int sample1DPos = sample2DPos.y * numCols + sample2DPos.x;
			
			color += weights[j + blurRadius] * weights[i + blurRadius] * d_color[sample1DPos];
		}
	}
	//color /= (2 * blurRadius + 1) * (2 * blurRadius + 1);


	d_colorBlurred[thread_1D_pos] = (unsigned char)color;
}