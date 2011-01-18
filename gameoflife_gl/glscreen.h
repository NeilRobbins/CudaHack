#include <stdlib.h>


void initGlApp(unsigned int width, unsigned int height, void (*renderCallback)(void));
void lockTarget(unsigned int** pTarget);
void unlockTarget(unsigned int* pTarget);


