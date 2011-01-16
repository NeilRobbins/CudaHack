#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 150
#define HEIGHT 150
#define GRID_SIZE WIDTH * HEIGHT
#define ACTUAL_GRID_SIZE sizeof(char) * GRID_SIZE
#define BLOCK_WIDTH 1
#define BLOCK_HEIGHT 1

__global__ void runGeneration(char currentModel[], char nextModel[]);
__device__ void getBlock(int x, int y, int* topLeftXOfBlock, int* topLeftYOfBlock);
__device__ void getCountOfNeighbours(char model[], int x, int y, int* neighbourCount);
__device__ void addCellValue(char model[], int x, int y, int* value);
__device__ void runRules(char model[], int x, int y, char* fate);

void printGrid(char grid[]);

int main(char** args, int argCount) {
    char clientModel[GRID_SIZE];
    char* deviceModel = 0;
    char* deviceModel2 = 0;
       
    for(int index = 0 ; index < GRID_SIZE ; index++){
    	clientModel[index] = rand() % 2;
    }

    cudaMalloc( (void**)&deviceModel, ACTUAL_GRID_SIZE);
    cudaMalloc( (void**)&deviceModel2, ACTUAL_GRID_SIZE);
    dim3 grid(WIDTH, HEIGHT);


    cudaMemcpy( deviceModel, clientModel, ACTUAL_GRID_SIZE, cudaMemcpyHostToDevice);

    runGeneration<<<grid,1>>>(deviceModel, deviceModel2);

    cudaMemcpy( clientModel, deviceModel2, ACTUAL_GRID_SIZE, cudaMemcpyDeviceToHost );


    printGrid(clientModel);

    cudaFree(deviceModel);
    cudaFree(deviceModel2);
}

void printGrid(char grid[]) {
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            printf("%d", grid[x*y]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void runGeneration(char currentModel[], char nextModel[]) {
    int startx, starty;
    getBlock(blockIdx.x, blockIdx.y, &startx, &starty);
    for (int x = startx; x < startx + BLOCK_WIDTH; x++) {
        for (int y = starty; y < starty + BLOCK_HEIGHT; y++) {
            int index = x + (y * WIDTH);
            runRules(currentModel, x, y, &nextModel[index]);
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
    if (x > WIDTH || y > HEIGHT) return;

    int index = x + (y * WIDTH);
    *value += model[index];
}

__device__ void runRules(char model[], int x, int y, char* fate) {
    int count;
    getCountOfNeighbours(model, x, y, &count);
    int index = x + (y * WIDTH);
    if (model[index] == 0) {
        if (count < 2 || count > 3) *fate = 0;
        if (count == 2 || count == 3) *fate = 1;
    } else {
        if (count == 3) { *fate = 1; }
        else { *fate = 0; }
    } 
}


