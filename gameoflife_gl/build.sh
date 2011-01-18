nvcc --compiler-options -fpermissive -c main.cu glscreen.cu -I.
nvcc -o main main.o glscreen.o  -lGL -lGLU -lX11 -lXi -lXmu -lGLEW -lglut
