#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
using namespace std;


__global__ void Add(int *a, int* b, int* c)
{
	printf("a=%d\n", *a);
	printf("b=%d\n", *b);
	*c = (*a) + (*b);
}

void FirstTask()
{
	int a, b, c; // on host
	cin >> a >> b;
	//cout << "b = " << b << endl;
	//cout << "a = " << a << endl;
	int *devA, *devB, *devC;
	//memory on dev
	cudaMalloc((void**)&devA, sizeof(int));
	cudaMalloc((void**)&devB, sizeof(int));
	cudaMalloc((void**)&devC, sizeof(int));
	//copy host to device
	cudaMemcpy(devA, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, &b, sizeof(int), cudaMemcpyHostToDevice);
	Add << <2, 2 >> > (devA, devB, devC);
	//copy of the result from device to host
	cudaMemcpy(&c, devC, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d + %d = %d\n", a, b, c);
}

void SecondTask()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device name : %s\n", deviceProp.name);
	printf("Total global memory : %d MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
	printf("Memory Bus Width: %d\n", deviceProp.memoryBusWidth);
	printf("Shared memory per block : %d\n", deviceProp.sharedMemPerBlock);
	printf("Registers per block : %d\n", deviceProp.regsPerBlock);
	printf("Warp size : %d\n", deviceProp.warpSize);
	printf("Memory pitch : %d\n", deviceProp.memPitch);
	printf("Max threads per block : %d\n", deviceProp.maxThreadsPerBlock);
	printf("Max threads dimensions : x = %d, y = %d, z =%d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("Max grid size: x = %d, y = %d, z = %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Clock rate: %d\n", deviceProp.clockRate);
	printf("Total constant memory: %d\n", deviceProp.totalConstMem);
	printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("Texture alignment: %d\n", deviceProp.textureAlignment);
	printf("Device overlap: %d\n", deviceProp.deviceOverlap);
	printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
	printf("Kernel execution timeout enabled: %s\n", deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
}

int main()
{
	FirstTask();
	SecondTask();

	cudaDeviceSynchronize();
	system("Pause");
	return 0;
}