#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_gl_interop.h>
#include "../common/cutil.h"

const int SCREENWIDTH = 512;
const int SCREENHEIGHT = 512;

int g_phase = 0;

__global__ void device_render(unsigned int* output, int width, int height, int phase);


GLuint pixelBufferObject;
struct cudaGraphicsResource *cudaPixelBufferObject;

void initGL(int argc, char** args);


void idleCallback();
void reshapeCallback(int width, int height);
void displayCallback();

void createBuffers();
void releaseBuffers();


int main(int argc, char** args){
	initGL(argc, args);

	char* pBuffer;

	// Just use the first device we find
	// NOTE: Found this out hte hard way, we HAVE to do this
	// In order to use interopability
	cudaSetDevice(0);
    	cudaGLSetGLDevice(0);


	cudaMalloc( (void**)&pBuffer, sizeof(char) * 512 * 512);

	createBuffers();
	atexit(releaseBuffers);
	glutMainLoop();
	cudaThreadExit();
	return 0;
}


void initGL(int argc, char** args){
	glutInit(&argc, args);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(SCREENWIDTH, SCREENHEIGHT);
	glutCreateWindow("Test GL App");
	glutDisplayFunc(displayCallback);
	glutReshapeFunc(reshapeCallback);
	glutIdleFunc(idleCallback);
	glewInit();
}

void reshapeCallback(int width, int height){
	glViewport(0,0,width,height);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

}

void displayCallback()
{
	unsigned int* output;
	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cudaPixelBufferObject, 0));
	size_t num_bytes;
	CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer( (void**)&output, &num_bytes, cudaPixelBufferObject));

	dim3 blockSize(16,16, 1);
	dim3 gridSize(SCREENWIDTH / blockSize.x, SCREENHEIGHT / blockSize.y);
	device_render<<<gridSize, blockSize>>>(output, SCREENWIDTH, SCREENHEIGHT,g_phase++ );

	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cudaPixelBufferObject,0 ));

	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glRasterPos2i(0,0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pixelBufferObject);
	glDrawPixels(SCREENWIDTH, SCREENHEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glutSwapBuffers();	
	glutReportErrors();
}

void idleCallback(){
	printf(" *Tick*\n");	
	glutPostRedisplay();
}

void createBuffers(){

	// All of this sets up the pixel buffer object
	glGenBuffersARB(1, &pixelBufferObject);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pixelBufferObject);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 
		SCREENWIDTH * SCREENHEIGHT * sizeof(GLubyte) * 4,
		0,
		GL_STREAM_DRAW_ARB);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// And this then allows CUDA/GL interop on that object
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cudaPixelBufferObject, pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard));

}

void releaseBuffers(){
	cudaGraphicsUnregisterResource(cudaPixelBufferObject);
	glDeleteBuffersARB(1, &pixelBufferObject);
}


__global__ void device_render(unsigned int* output, int width, int height, int phase)
{
	unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	output[x + y * width] = (x  * y + phase);

}
























