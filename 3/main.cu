#include <stdio.h>

#define N 10
#define NN 100


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
	int clientInput[NN];
	int clientOutput[NN];
	int* gpuInput;
	int* gpuOutput;

	for(int x =0; x < N ; x++){
		for(int y =0 ; y < N ; y++){
			clientInput[x + y * N] = x + y;
		}
	}


	dim3 grid(N,N);

	cudaMalloc( (void**)&gpuInput, NN * sizeof(int));
	cudaMalloc( (void**)&gpuOutput, NN * sizeof(int));

	cudaMemcpy( gpuInput, clientInput, NN * sizeof(int), cudaMemcpyHostToDevice ); 

	// Operation here - X, Y????
	grid_sum<<<grid, 1>>>(gpuInput, gpuOutput);

	cudaMemcpy(  clientOutput, gpuOutput, NN * sizeof(int), cudaMemcpyDeviceToHost );

	for(int x = 0; x < N ; x++){
		for(int y = 0 ; y < N ; y++){
			printf(" %d ", clientInput[x + y * N]);
		}
		printf("\n");
	}

	for(int x = 0 ; x < N ; x++){
		for(int y = 0 ; y < N ; y++){
			printf(" %d ", clientOutput[x + y * N]);
		}
		printf("\n");
	}
	
	
	cudaFree(gpuInput);
	cudaFree(gpuOutput);

}
