#include <stdio.h>

__global__ void multiplyNumbersByAScalar(float numbers[], float scalar) {
	int x = blockIdx.x;
	numbers[x] = numbers[x] * scalar;
}


int main(int argc, char** args)
{
	float numbersInSystemMemory[] = { 0, 1, 2 , 3 , 4 , 5 , 6 ,7 ,8 , 9};
	float* numbersInDeviceMemory;

	cudaMalloc( (void**)&numbersInDeviceMemory, sizeof(float) * 10);
	
	cudaMemcpy( numbersInDeviceMemory, numbersInSystemMemory, sizeof(float) * 10, cudaMemcpyHostToDevice ); 
	
	multiplyNumbersByAScalar<<<10,1>>>(numbersInDeviceMemory, 2.0f);

	cudaMemcpy(  numbersInSystemMemory, numbersInDeviceMemory, sizeof(float) * 10, cudaMemcpyDeviceToHost ); 

	cudaFree( numbersInDeviceMemory );

	for(int x = 0; x < 10 ; x++){
		printf("%f ", numbersInSystemMemory[x]);
	}


	return 1;
}
