
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




__global__ void ScalMult(const float *a, const
	float *b, float *c, int numElem)
{
	// Переменная для хранения суммы элементов

	// Создание массивов в разделяемой памяти
	__shared__ float arrShared[THREADS_PER_BLOCK];
	// Копирование из глобальной памяти

	printf("----------------------\n");
	printf("a[%d * %d + %d] = %f\n", blockIdx.x, blockDim.x, threadIdx.x, a[blockIdx.x * blockDim.x + threadIdx.x]);
	printf("b[%d * %d + %d] = %f\n", blockIdx.x, blockDim.x, threadIdx.x, b[blockIdx.x * blockDim.x + threadIdx.x]);
	arrShared[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x] * b[blockIdx.x * blockDim.x + threadIdx.x];
	// Синхронизация нитей
	__syncthreads();
	// Вычисление скалярного произведения
	if (threadIdx.x == 0)
	{
		float sum = 0.0;
		for (int j = 0; j < THREADS_PER_BLOCK; j++)
		{
			sum += arrShared[j];
		}
		/*c[blockIdx.x] = sum;*/
		atomicAdd(c, sum);
	}
}

__global__ void eval_sqrtf(float *x, float *res)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx<N)
	{
		*res = sqrtf(*x);
	}

}

__global__ void eval_x(float *x, float * h)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N)
	{
		x[idx] = idx * (*h);
		//printf("x[%i] = %f\n", idx, x[idx]);
	}
}

__global__ void eval_y_rect(float *x, float *res)
{
	__shared__ float temp[THREADS_PER_BLOCK];
	temp[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		float sum = 0.0;
		for (int j = 0; j < THREADS_PER_BLOCK - 1; j++)
		{
			sum += sqrtf(1 - powf((temp[j] + temp[j + 1]) / 2.0, 2))*(temp[j + 1] - temp[j]);
			//printf("y[] = %f\n", sqrtf(1 - powf((temp[j] + temp[j + 1]) / 2.0, 2))*(temp[j + 1] - temp[j]) );
		}
		atomicAdd(res, sum);
	}
}

__global__ void eval_y_trapz(float *x, float *res)
{
	__shared__ float temp[THREADS_PER_BLOCK];
	temp[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		float sum = 0.0;
		for (int j = 0; j < THREADS_PER_BLOCK-1; j++)
		{
			sum += (sqrtf(1 - powf(temp[j], 2)) + sqrtf(1 - powf(temp[j + 1], 2))) / 2.0 * (temp[j + 1] - temp[j]);
			//printf("y[] = %f\n", sqrtf(1 - powf((temp[j] + temp[j + 1]) / 2.0, 2))*(temp[j + 1] - temp[j]));
		}
		atomicAdd(res, sum);
	}
}

__global__ void eval_y_simps(float *x, float *h, float *res)
{
	__shared__ float temp[THREADS_PER_BLOCK];
	temp[threadIdx.x] = x[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		float sum = 0.0;
		for (int j = 1; j < THREADS_PER_BLOCK - 1; j = j + 2)
		{
			sum += (*h / 3.0 * (sqrtf(1 - powf(temp[j - 1], 2)) + 4.0*sqrtf(1 - powf(temp[j], 2)) + sqrtf(1 - powf(temp[j - 1], 2))));
			//printf("y[] = %f\n", sqrtf(1 - powf((temp[j] + temp[j + 1]) / 2.0, 2))*(temp[j + 1] - temp[j]));
		}
		atomicAdd(res, sum);
	}
}


void FirstLab()
{
	// вычисления на хосте
	float *h_a;
	h_a = (float*)malloc(N * sizeof(float));
	float *h_b;
	h_b = (float *)malloc(N * sizeof(float));
	float h_c = 0;
	for (int i = 0; i< N; i++)
	{
		h_a[i] = rand() % 10 + 1;
		printf("a[%d] = %f\n", i, h_a[i]);
		h_b[i] = rand() % 10 + 1;
		printf("b[%d] = %f\n", i, h_b[i]);
	}
	printf("Arrays' size = %i\n", N);
	for (int i = 0; i != N; i++)
	{
		float temp = h_a[i] * h_b[i];
		h_c += temp;
	}

	printf("host result = %f\n", h_c);

	h_c = 0;
	// на девайсе
	float * d_a;
	cudaMalloc((void**)&d_a, N * sizeof(float));
	float * d_b;
	cudaMalloc((void**)&d_b, N * sizeof(float));
	float *d_c;
	cudaMalloc((void**)&d_c, sizeof(float));
	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice);
	ScalMult << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_a, d_b, d_c, N);
	//float *h_cc = (float*)malloc(sizeof(float));
	//cudaMemcpy(h_cc, &d_c,  sizeof(float), cudaMemcpyDeviceToHost);
	float res = 0;

	cudaMemcpy(&res, d_c, sizeof(float), cudaMemcpyDeviceToHost);
	/*for (int i = 0; i != N; i++)
	{
		res += h_cc[i];
	}*/
	printf("device result= %f\n", res);


}

void FirstHome()
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
	cudaMalloc((void**)&d_x, sizeof(float)*N);
	cudaMalloc((void**)&d_y, sizeof(float)*N);

	float res1 = 0;
	float * d_res1;
	cudaMalloc((void**)&d_res1, sizeof(float));
	cudaMemcpy(d_res1, &res1, sizeof(float), cudaMemcpyHostToDevice);
	float res2 = 0;
	float * d_res2;
	cudaMalloc((void**)&d_res2, sizeof(float));
	cudaMemcpy(d_res2, &res2, sizeof(float), cudaMemcpyHostToDevice);
	float res3 = 0;
	float * d_res3;
	cudaMalloc((void**)&d_res3, sizeof(float));
	cudaMemcpy(d_res3, &res3, sizeof(float), cudaMemcpyHostToDevice);
	float h = (b - a)*1.0 / (N);
	printf("h = %f\n", h);
	float * d_h;
	cudaMalloc((void**)&d_h, sizeof(float));
	cudaMemcpy(d_h, &h, sizeof(float), cudaMemcpyHostToDevice);

	eval_x << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_x, d_h);
	eval_y_rect << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_x, d_res1);
	cudaMemcpy(&res1, d_res1, sizeof(float), cudaMemcpyDeviceToHost);
	eval_y_trapz << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_x, d_res2);
	cudaMemcpy(&res2, d_res2, sizeof(float), cudaMemcpyDeviceToHost);
	eval_y_simps << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_x, d_h, d_res3);
	cudaMemcpy(&res3, d_res3, sizeof(float), cudaMemcpyDeviceToHost);
	res1 *= 4.0;
	res2 *= 4.0;
	res3 *= 4.0;
	printf("res rect = %f\n", res1);
	printf("res trapz = %f\n", res2);
	printf("res simpson = %f\n", res3);
	free(x); free(y);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_h); cudaFree(d_res1); cudaFree(d_res2); cudaFree(d_res3);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("%s", cudaGetErrorString(err));
}

void SecondHome()
{
	float *x;
	float *d_x;
	x = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		x[i] = rand() / (float)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	cudaMalloc((void**)&d_x, sizeof(float)*N);
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

	float *h_res1 = (float*)malloc(sizeof(float));
	float * d_res1; float * dd_res1;
	cudaMalloc((void**)&d_res1, N * sizeof(float));
	cudaMalloc((void**)&dd_res1, sizeof(float));
	ScalMult << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_x, d_x, d_res1, N);
	eval_sqrtf << <1, 1 >> > (d_res1, dd_res1);
	cudaMemcpy(h_res1, dd_res1, sizeof(float), cudaMemcpyDeviceToHost);

	float sum = 0;
	for (int i = 0; i < N; ++i) {
		sum += x[i] * x[i];
	}

	sum = sqrtf(sum);
	printf("sum  = %f\n", sum);
	//res1 = powf(res1,0.5);

	printf("res  = %f\n", *h_res1);
	free(x);
	cudaFree(d_res1);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("%s", cudaGetErrorString(err));
}

int main()
{
	//FirstLab();
	//FirstHome();
	SecondHome();
	cudaDeviceSynchronize();
	getchar();
	return 0;
}
