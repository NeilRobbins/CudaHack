#include <stdlib.h>

void setupGlApp(unsigned int width, unsigned int height);
void runGlApp(void (*renderCallback)(void));
void lockTarget(unsigned int** pTarget);
void unlockTarget(unsigned int* pTarget);


