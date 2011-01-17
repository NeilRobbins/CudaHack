#include <stdio.h>

// So, a __global__ to tell NVCC this is a device function for calling from the host
// and a pointer to some memory to write the result to
__global__ void addTwoNumbers(int x, int y, int *result) {
	*result = x + y;
}


int main(int argc, char** args)
{
	// This is in system memory
	int cpuVisibleResult;

	// This is just an unitialized pointer
	int *gpuVisibleResult;

	// This sets that pointer to point at a memory location on device memory
	cudaMalloc( (void**)&gpuVisibleResult, sizeof(int) );

	// Call the method, ignore 1,1 for now
	addTwoNumbers<<<1,1>>>(2,7, gpuVisibleResult);

	// Download the result from the device to the host
	cudaMemcpy( &cpuVisibleResult, gpuVisibleResult, sizeof(int), cudaMemcpyDeviceToHost );

	// Print the results
	printf( " 2 + 7 = %d\n", cpuVisibleResult);

	// Free up that memory on the device
	cudaFree( gpuVisibleResult );

	return 1;
}
