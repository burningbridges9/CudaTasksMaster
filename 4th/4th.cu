
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
using namespace std;
#define N 1024
#define THREADS_PER_BLOCK 256
#define BLOCK_NUM  16

__global__ void Eval(int *nCirc, float * x, float *y)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < N)
	{
		if (x[i] * x[i] + y[i] * y[i] <= 1)
		{
			atomicAdd(nCirc, 1);
		}
	}
}

__global__ void Evalx(float * x, unsigned seed)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	curandState_t t;
	curand_init(seed, i, 0, &t);
	if (i < N)
	{
		x[i] = curand_uniform(&t);
		//printf("x[%i]=%f\n", i, x[i]);
	}
}

__global__ void Evaly(float * y, unsigned seed)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	printf("blockDim.x=%i\n", blockDim.x);
	printf("blockIdx.x=%i\n", blockIdx.x);
	curandState_t m;
	curand_init(seed, i, 0, &m);
	if (i < N)
	{
		y[i] = curand_uniform(&m);
		//printf("y[%i]=%f\n", i, y[i]);
	}
}

void PiCalculation()
{
	int sizeI = sizeof(int);
	int sizeF = sizeof(float);
	int nCirc = 0;
	float x[N];
	float y[N];
	int *devNcirc;
	float *devx, *devy;
	cudaMalloc((void**)&devx, N*sizeF);
	cudaMalloc((void**)&devy, N*sizeF);
	cudaMalloc((void**)&devNcirc, sizeI);
	cudaMemcpy(devNcirc, &nCirc, sizeI, cudaMemcpyHostToDevice);
	Evalx << <BLOCK_NUM, THREADS_PER_BLOCK >> > (devx, 0);
	Evaly << <BLOCK_NUM, THREADS_PER_BLOCK >> > (devy, time(NULL));
	cudaMemcpy(&x, devx, N*sizeF, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y, devy, N*sizeF, cudaMemcpyDeviceToHost);
	/*for (int i = 0; i != N; i++)
	{
	printf("x[%i]=%f\n", i, x[i]);
	printf("y[%i]=%f\n", i, y[i]);
	}*/
	Eval << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (devNcirc, devx, devy);
	cudaMemcpy(&nCirc, devNcirc, sizeI, cudaMemcpyDeviceToHost);
	printf("pi = %f\n", (nCirc*4.0) / N);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s ",
		cudaGetErrorString(err));

	cudaFree(devNcirc);
	cudaFree(devx);
	cudaFree(devy);
}

int main()
{
	PiCalculation();
	cudaDeviceSynchronize();
	getchar();
}
