#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "glscreen.h"


const int SCREENWIDTH = 512;
const int SCREENHEIGHT = 512;

void renderCallback();

int main(int argc, char** args){
	
	// Sets up a window, texture, loop, etc
	initGlApp(SCREENWIDTH, SCREENHEIGHT, renderCallback);
	
	return 0;
}

void renderCallback(){
	
	unsigned int* pTarget;
	lockTarget(&pTarget);

	// Call device function against target

	unlockTarget(pTarget);

}
























