#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
using namespace std;


__global__ void Add(int *a, int* b, int* c)
{
	printf("a=%d", *a);
	printf("b=%d", *b);
	*c = (*a) + (*b);
}

int main()
{
	//HelloWorld <<< 2, 5 >>>();
	int a, b, c; // on host
	cin >> a >> b;
	cout << "a = " << a << endl;
	cout << "b = " << b << endl;
	int *devA, *devB, *devC;
	//memory on dev
	cudaMalloc((void**)&devA, sizeof(int));
	cudaMalloc((void**)&devB, sizeof(int));
	cudaMalloc((void**)&devC, sizeof(int));
	//copy host to device
	cudaMemcpy(devA, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, &b, sizeof(int), cudaMemcpyHostToDevice);
	Add << <1, 1 >> > (devA, devB, devC);
	//copy of the result from device to host
	cudaMemcpy(&c, devC, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d + %d = %d\n", a, b, c);
	cudaDeviceSynchronize();
	system("Pause");
	return 0;
}