
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
//#define DOTS_NUM 1024 //chislo tochek v kvadrate

#pragma region Pi


#define N 128
#define THREADS_PER_BLOCK 64
#define BLOCK_NUM  16
__global__ void eval_x(float *x, float * h)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N)
	{
		x[idx] = idx * (*h);
		printf("x[%i] = %f\n", idx, x[idx]);
	}
}

__global__ void eval_y(float *x, float *y, float *h)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N)
	{
		y[idx] = (*h) * sqrtf(1 - x[idx] * x[idx]);
		printf("y[%i] = %f\n", idx, y[idx]);
	}
}

void PiCalculation()
{
	int a = 0, b = 1;
	int *d_a, *d_b;
	cudaMalloc((void**)&d_a, sizeof(int));
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

	float *x, *y;
	float *d_x, *d_y;
	x = (float*)malloc(sizeof(float)*N);
	y = (float*)malloc(sizeof(float)*N);
	cudaMalloc((void**)&d_x, sizeof(int)*N);
	cudaMalloc((void**)&d_y, sizeof(int)*N);

	float res = 0;

	float h = (b - a)*1.0 / N;
	printf("h = %f\n", h);
	float * d_h;
	cudaMalloc((void**)&d_h, sizeof(float));
	cudaMemcpy(d_h, &h, sizeof(float), cudaMemcpyHostToDevice);

	eval_x << <BLOCK_NUM, THREADS_PER_BLOCK >> > (d_x, d_h);
	eval_y << <BLOCK_NUM, THREADS_PER_BLOCK >> > (d_x, d_y, d_h);
	cudaMemcpy(y, d_y, sizeof(float)*N, cudaMemcpyDeviceToHost);

	for (int i = 0; i != N; i++)
	{
		res += y[i];
	}
	res *= 4.0;
	printf("res = %f\n", res);

	free(x); free(y);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_h);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("%s", cudaGetErrorString(err));
}

#pragma endregion


#pragma region Zeta function

#define N 32
#define THREADS_PER_BLOCK 16
#define BLOCK_NUM  16

__global__ void ZetaFunc(float * s, float *res)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	printf("idx = %i\n", idx);
	if ((idx < N) && (idx != 0))
	{
		res[idx] = 1.0 / powf(idx, *s);
		printf("res[%i] = %f\n", idx, res[idx]);
	}
}


void ZetaFuncCalculation()
{
	float s;
	float *res;
	float *devS, *devRes;
	size_t size = sizeof(float);
	cudaMalloc((void**)&devS, size);
	cudaMalloc((void**)&devRes, size*N);
	s = 2;
	res = (float*)malloc(size*N);
	cudaMemcpy(devS, &s, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_res, res, size, cudaMemcpyHostToDevice);
	ZetaFunc << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (devS, devRes);
	cudaMemcpy(res, devRes, size*N, cudaMemcpyDeviceToHost);
	float r = 0;
	for (int i = 0; i != N; i++)
	{
		r += res[i];
	}
	printf("zeta = %f\n", r);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("%s ", cudaGetErrorString(err));
	free(res);
	cudaFree(devS);
	cudaFree(devRes);
}

#pragma endregion



int main()
{
	PiCalculation();
	//ZetaFuncCalculation();
	cudaDeviceSynchronize();
	getchar();
}
