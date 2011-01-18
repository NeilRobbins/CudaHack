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





void initGL(int argc, char** args);


void reshapeCallback(int width, int height);
void displayCallback();

void createBuffers();
void releaseBuffers();


int main(int argc, char** args){
	initGL(argc, args);

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

	glutSwapBuffers();	
}

