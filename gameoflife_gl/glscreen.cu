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


GLuint g_screenPixelBuffer;
struct cudaGraphicsResource* g_cudaScreenPixelBuffer = 0;
void (*g_renderCallback)(void);
unsigned int g_width;
unsigned int g_height;

void idleCallback();
void reshapeCallback(int width, int height);
void displayCallback();

void createBuffers();
void releaseBuffers();

void initGlApp(unsigned int width, unsigned int height, void (*renderCallback)(void)){
        
	int argc = 0;
	char** args = 0;
	g_width = width;
	g_height = height;

	glutInit(&argc, args);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
        glutInitWindowSize(g_width, g_height);
        glutCreateWindow("GL Screen");
        glutDisplayFunc(displayCallback);
        glutReshapeFunc(reshapeCallback);
        glutIdleFunc(idleCallback);
        glewInit();


        cudaSetDevice(0);
        cudaGLSetGLDevice(0);

        createBuffers();
        atexit(releaseBuffers);
        glutMainLoop();
        cudaThreadExit();
	
}

void lockTarget(unsigned int** pTarget){

}

void unlockTarget(unsigned int* pTarget){


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
//	g_renderCallback();

        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glRasterPos2i(0,0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, g_screenPixelBuffer);
        glDrawPixels(g_width, g_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        glutSwapBuffers();
        glutReportErrors();
}

void idleCallback(){
        glutPostRedisplay();
}

void createBuffers(){

        // All of this sets up the pixel buffer object
        glGenBuffersARB(1, &g_screenPixelBuffer);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, g_screenPixelBuffer);
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                g_width * g_height * sizeof(GLubyte) * 4,
                0,
                GL_STREAM_DRAW_ARB);

        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        // And this then allows CUDA/GL interop on that object
        CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&g_cudaScreenPixelBuffer, g_screenPixelBuffer, cudaGraphicsMapFlagsWriteDiscard));

}

void releaseBuffers(){
        cudaGraphicsUnregisterResource(g_cudaScreenPixelBuffer);
        glDeleteBuffersARB(1, &g_screenPixelBuffer);
}



