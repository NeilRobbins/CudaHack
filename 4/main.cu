#include <stdio.h>

#define N 100
#define NN 1000e


__device__ void add_value(int* input, int x, int y, int* output);

__global__ void grid_sum(int* input, int* output) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	if(x >= N) { return; }
	if(y >= N) { return; }

	output[x + y * N] = 0;
		add_value(input, x-1, y, &output[x + y * N]); 
		add_value(input, x+1, y, &output[x + y * N]); 
		add_value(input, x, y-1, &output[x + y * N]); 
		add_value(input, x, y+1, &output[x + y * N]);

}

__device__ void add_value(int* input, int x, int y, int* output) {
	if(x < 0) { return;}
	if(y < 0) { return;}
	if(y >= N) { return; }
	if(x >= N) { return; }
	
	*output +=  input[x + y * N];

}


int main(char** args, int argc)
{

	printf("Declaring variables");

	int clientInput[NN];
	int clientOutput[NN];
	int* gpuInput;
	int* gpuOutput;

	printf("About to initalize data");

	for(int x =0; x < N ; x++){
		for(int y =0 ; y < N ; y++){
			clientInput[x + y * N] = x + y;
		}
	}

	printf("Data initialized");

	dim3 grid(N,N);

	cudaMalloc( (void**)&gpuInput, NN * sizeof(int));
	cudaMalloc( (void**)&gpuOutput, NN * sizeof(int));

	for(int x = 0 ; x < 10000 ; x++) {

	cudaMemcpy( gpuInput, clientInput, NN * sizeof(int), cudaMemcpyHostToDevice ); 

	// Operation here - X, Y????
	grid_sum<<<grid, 1>>>(gpuInput, gpuOutput);

	


	cudaMemcpy(  clientOutput, gpuOutput, NN * sizeof(int), cudaMemcpyDeviceToHost );
	
	printf("Loop %d completed\n", x);

	}

	
	cudaFree(gpuInput);
	cudaFree(gpuOutput);


}
