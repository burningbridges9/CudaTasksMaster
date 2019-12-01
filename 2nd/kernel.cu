
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
using namespace std;

#define BLOCK_SIZE 32
// тип, который будут иметь элементы матриц
#define BASE_TYPE float
#define THREADS_PER_BLOCK 128
// размер 
#define N BLOCK_SIZE * THREADS_PER_BLOCK


__constant__ float constDataA[N];
__constant__ float constDataB[N];



__global__ void scalMult2(BASE_TYPE *C)
{
	__shared__ BASE_TYPE ash[BLOCK_SIZE];
	__shared__ BASE_TYPE bsh[BLOCK_SIZE];
	ash[threadIdx.x] = constDataA[blockIdx.x * blockDim.x + threadIdx.x] * constDataB[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		BASE_TYPE sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
		{
			sum += ash[j];
		}
		C[blockIdx.x] = sum;
	}
}

int main()
{
	BASE_TYPE *h_a;
	h_a = (BASE_TYPE*)malloc(N * sizeof(BASE_TYPE));
	BASE_TYPE *h_b;
	h_b = (BASE_TYPE *)malloc(N * sizeof(BASE_TYPE));
	BASE_TYPE h_c = 0;
	for (int i = 0; i< N; i++)
	{
		h_a[i] = 1;// rand() % 10 + 1;
				   //printf("h_a[%f] = %f\n", i, h_a[i]);
		h_b[i] = 1;// rand() % 10 + 1;
				   //printf("h_b[%f] = %f\n", i, h_b[i]);
	}
	printf("scalar on host:\n");
	for (int i = 0; i != N; i++)
	{
		BASE_TYPE temp = h_a[i] * h_b[i];
		h_c += temp;
	}
	printf("host result = %f\n", h_c);

	h_c = 0;

	BASE_TYPE * d_c;
	cudaMalloc((void**)&d_c, N * sizeof(BASE_TYPE));
	cudaMemcpy(d_c, &h_c, sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	// копирование данных с центрального процессора в
	// константную память
	cudaMemcpyToSymbol(constDataA, h_a, N * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constDataB, h_b, N * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	scalMult2 << < BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_c);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	BASE_TYPE *h_cc = (BASE_TYPE*)malloc(N * sizeof(BASE_TYPE));
	cudaMemcpy(h_cc, d_c, N * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);


	for (int i = 1; i != N; i++)
	{
		h_cc[0] += h_cc[i];
	}
	printf("h_cc[%i] = %f\n", 0, h_cc[0]);

	cudaDeviceSynchronize();
	getchar();
	return 0;
}
