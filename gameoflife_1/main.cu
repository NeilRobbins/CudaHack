#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 1000
#define HEIGHT 1000
#define GRID_SIZE WIDTH * HEIGHT
#define ACTUAL_GRID_SIZE sizeof(char) * GRID_SIZE
#define BLOCK_WIDTH 1
#define BLOCK_HEIGHT 1
#define NO_OF_GENERATIONS_TO_RUN 5000

//#define DUMPFULL
#define DUMPCOUNT
//#define CUBE


__global__ void runGeneration(char currentModel[], char nextModel[]);
__device__ void getBlock(int x, int y, int* topLeftXOfBlock, int* topLeftYOfBlock);
__device__ void getCountOfNeighbours(char model[], int x, int y, int* neighbourCount);
__device__ void addCellValue(char model[], int x, int y, int* value);
__device__ void runRules(char model[], int x, int y, char* fate);

void printGrid(char grid[]);

int main(char** args, int argCount) {
    printf("%d;%d\n", WIDTH, HEIGHT);
    srand( time(NULL) );

     char clientModel[GRID_SIZE];
    char* deviceModel = 0;
    char* deviceModel2 = 0;
       
    for(int index = 0 ; index < GRID_SIZE ; index++){
    	clientModel[index] =  rand() % 2;
    }

    printGrid(clientModel);

    cudaMalloc( (void**)&deviceModel, ACTUAL_GRID_SIZE);
    cudaMalloc( (void**)&deviceModel2, ACTUAL_GRID_SIZE);
    dim3 grid(WIDTH, HEIGHT);

    cudaMemcpy( deviceModel, clientModel, ACTUAL_GRID_SIZE, cudaMemcpyHostToDevice);

    char* swapPointer = 0;
    for (int generation = 0; generation < NO_OF_GENERATIONS_TO_RUN; generation++) {
	runGeneration<<<grid,1>>>(deviceModel, deviceModel2);
	swapPointer = deviceModel;
	deviceModel = deviceModel2;
	deviceModel2 = swapPointer;
    
    	cudaMemcpy( clientModel, deviceModel, ACTUAL_GRID_SIZE, cudaMemcpyDeviceToHost );
    	printGrid(clientModel);
    }

    cudaFree(deviceModel);
    cudaFree(deviceModel2);
}

void printGrid(char grid[]) {
#ifdef CUBE
    for(int y = 0; y < HEIGHT ; y++) {
       	for(int x = 0; x < WIDTH; x++) {
	    printf("%d", grid[x + y * WIDTH]);
	}
	printf("\n");	
    }
    printf("\n");
#endif

#ifdef DUMPFULL 

    for(int y = 0; y < HEIGHT ; y++) {
       	for(int x = 0; x < WIDTH; x++) {
	    printf("%d", grid[x + y * WIDTH]);
	}
    }

#endif

#ifdef DUMPCOUNT

    int count = 0;
    for(int x = 0; x < GRID_SIZE ; x++){
	if(grid[x] == 1) count++;
    }
    printf("%d\n", count);

#endif

}

__global__ void runGeneration(char currentModel[], char nextModel[]) {
    int startx, starty;

    getBlock(blockIdx.x, blockIdx.y, &startx, &starty);
    for (int x = startx; x < startx + BLOCK_WIDTH; x++) {
        for (int y = starty; y < starty + BLOCK_HEIGHT; y++) {
            int index = x + (y * WIDTH);
            runRules(currentModel, x, y, nextModel + index);
        }
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
    if (x < 0 || y < 0) return;
    if (x >= WIDTH || y >= HEIGHT) return;
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
