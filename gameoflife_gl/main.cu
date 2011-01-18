#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <driver_functions.h>
#include <cuda_gl_interop.h>

#include "glscreen.h"

__global__ void runGeneration(unsigned int* pTarget, char currentModel[], char nextModel[]);
__device__ void getBlock(int x, int y, int* topLeftXOfBlock, int* topLeftYOfBlock);
__device__ void getCountOfNeighbours(char model[], int x, int y, int* neighbourCount);
__device__ void addCellValue(char model[], int x, int y, int* value);
__device__ void runRules(char model[], int x, int y, char* fate);

#define WIDTH 1760 
#define HEIGHT 1024
#define BLOCK_SIZE 16

#define GRID_SIZE WIDTH * HEIGHT
#define ACTUAL_GRID_SIZE GRID_SIZE * sizeof(char)


char* g_deviceModel1 = 0;
char* g_deviceModel2 = 0;


void doEverything(){
      	char* swapPointer = 0;
	unsigned int* pBuffer;

	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize(WIDTH / BLOCK_SIZE, HEIGHT / BLOCK_SIZE );


	lockTarget(&pBuffer);        

	// This will do all the logic and rendering in a single remote call
	runGeneration<<<gridSize,blockSize>>>(pBuffer, g_deviceModel1, g_deviceModel2);

	unlockTarget(pBuffer);

	// And now we swap the inputs
        swapPointer = g_deviceModel1;
        g_deviceModel1 = g_deviceModel2;
        g_deviceModel2 = swapPointer;
}

int main(int argc, char** args) {

	char g_clientModel[GRID_SIZE];
	cudaSetDevice(0);
        cudaGLSetGLDevice(0);

	setupGlApp(WIDTH, HEIGHT);


	srand( time(NULL) );
       
	for(int index = 0 ; index < GRID_SIZE ; index++){
		g_clientModel[index] =  rand() % 2;
    	}

    	cudaMalloc( (void**)&g_deviceModel1, ACTUAL_GRID_SIZE);
    	cudaMalloc( (void**)&g_deviceModel2, ACTUAL_GRID_SIZE);

    	cudaMemcpy( g_deviceModel1, g_clientModel, ACTUAL_GRID_SIZE, cudaMemcpyHostToDevice);


    	// Run the darned app
	runGlApp(doEverything);	


    	cudaFree(g_deviceModel1);
    	cudaFree(g_deviceModel2);
}



__global__ void runGeneration(unsigned int* pTarget, char currentModel[], char nextModel[]) {
	
	int x =  blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = x + (y * WIDTH);
        runRules(currentModel, x, y, nextModel + index);

	if(nextModel[index] == 1){
		pTarget[index] = 0xFFFFFFFF;
	}
	else
	{
		pTarget[index] = 0x00000000;
	}		

}

__device__ void getBlock(int x, int y, int* topLeftXOfBlock, int* topLeftYOfBlock) {
    *topLeftXOfBlock = x;
    *topLeftYOfBlock = y;
}

__device__ void getCountOfNeighbours(char model[], int x, int y, int* neighbourCount) {
   *neighbourCount = 0;
    addCellValue(model, x + 1, y, neighbourCount);
    addCellValue(model, x + 1, y + 1, neighbourCount);
    addCellValue(model, x + 1, y - 1, neighbourCount);
    addCellValue(model, x, y + 1, neighbourCount);
    addCellValue(model, x, y - 1, neighbourCount);
    addCellValue(model, x - 1, y, neighbourCount);
    addCellValue(model, x - 1, y + 1, neighbourCount);
    addCellValue(model, x - 1, y - 1, neighbourCount);
}

__device__ void addCellValue(char model[], int x, int y, int* value) {
    	if (x < 0) { x = WIDTH-1;}
	if( y < 0) { y = HEIGHT-1;}
	if( x >= WIDTH) { x = 0; }
	if( y >= HEIGHT) { y = 0; }
    	*value += model[x + y * WIDTH];
}

__device__ void runRules(char model[], int x, int y, char* fate) {
    int count;
    getCountOfNeighbours(model, x, y, &count);
    int index = x + (y * WIDTH);
    
    if (model[index] == 1) {
        if (count < 2 || count > 3) *fate = 0;
        if (count == 2 || count == 3) *fate = 1;
    } else {
        if (count == 3) { *fate = 1; }
        else { *fate = 0; }
    } 
}
