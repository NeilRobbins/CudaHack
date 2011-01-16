#include <stdio.h>

#define N 10

__global__ void pointless(int* input, int* output) {
	int x = blockIdx.x;
	if(x >= N) { return;}
	if(x == 0) { output[x] = input[x + 1]; return;}
	if(x == N-1) { output[x] = input[x-1]; return;}
	output[x] = input[x-1] + input[x + 1];

}




int main(char** args, int argc)
{
	int clientInput[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int clientOutput[N];
	int* gpuInput;
	int* gpuOutput;

	cudaMalloc( (void**)&gpuInput, N * sizeof(int));
	cudaMalloc( (void**)&gpuOutput, N * sizeof(int));

	cudaMemcpy( gpuInput, clientInput, N * sizeof(int), cudaMemcpyHostToDevice ); 

	// Operation here
	pointless<<<N, 1>>>(gpuInput, gpuOutput);

	cudaMemcpy(  clientOutput, gpuOutput, N * sizeof(int), cudaMemcpyDeviceToHost );

	for(int x = 0; x < N ; x++){
		printf(" %d ", clientInput[x]);
	}
	printf("\n");

	for(int x = 0 ; x < N ; x++){
		printf(" %d ", clientOutput[x]);
	}
	
	
	cudaFree(gpuInput);
	cudaFree(gpuOutput);

}
