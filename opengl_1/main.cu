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
//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
//#include <rendercheck_gl.h>


const int SCREENWIDTH = 800;
const int SCREENHEIGHT = 600;


GLuint pixelBufferObject;
struct cudaGraphicsResource *cudaPixelBufferObject;

void initGL(int argc, char** args);


void reshapeCallback(int width, int height);
void displayCallback();

void createBuffers();
void releaseBuffers();


int main(int argc, char** args){
	initGL(argc, args);
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
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);
	glRasterPos2i(0,0);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pixelBufferObject);
	glDrawPixels(SCREENWIDTH, SCREENHEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glutSwapBuffers();	
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
	cudaGraphicsGLRegisterBuffer(&cudaPixelBufferObject, pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);

}

void releaseBuffers(){
	cudaGraphicsUnregisterResource(cudaPixelBufferObject);
	glDeleteBuffersARB(1, &pixelBufferObject);
}



























