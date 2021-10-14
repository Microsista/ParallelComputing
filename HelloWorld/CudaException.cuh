#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

class CudaException : public std::runtime_error {
public:
	CudaException(const cudaError_t& cudaStatus, const char* file, unsigned int line) :
		runtime_error(CudaErrorToString(cudaStatus, file, line)) {}

private:
	std::string CudaErrorToString(const cudaError_t& cudaStatus, const char* file, unsigned int line) {
		char s_str[500]{};
		sprintf_s(s_str, "Exception thrown at line: %d, in file: %s, with message: %s.\n", line, file, cudaGetErrorString(cudaStatus));
		return std::string(s_str);
	}
};