
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



__global__ void ScalMult(BASE_TYPE *C)
{
	__shared__ BASE_TYPE ash[THREADS_PER_BLOCK];
	ash[threadIdx.x] = constDataA[blockIdx.x * blockDim.x + threadIdx.x] * constDataB[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		BASE_TYPE sum = 0.0;
		for (int j = 0; j < THREADS_PER_BLOCK; j++)
		{
			sum += ash[j];
		}
		atomicAdd(C, sum);
	}
}

void FirstLab()
{
	BASE_TYPE *h_a;
	h_a = (BASE_TYPE*)malloc(N * sizeof(BASE_TYPE));
	BASE_TYPE *h_b;
	h_b = (BASE_TYPE *)malloc(N * sizeof(BASE_TYPE));
	BASE_TYPE h_c = 0;
	for (int i = 0; i< N; i++)
	{
		h_a[i] = rand() / (BASE_TYPE)RAND_MAX;// rand() % 10 + 1;
				   //printf("h_a[%f] = %f\n", i, h_a[i]);
		h_b[i] = rand() / (BASE_TYPE)RAND_MAX;// rand() % 10 + 1;
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
	cudaMemcpyToSymbol(constDataA, h_a, N * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constDataB, h_b, N * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	ScalMult << < BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_c);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaMemcpy(&h_c, d_c, sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
	printf("h_c = %f\n", h_c);
}


void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

texture<float, 1, cudaReadModeElementType> texRefx;

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
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx + 1 <N)
	{
		//printf("temp[%i]-temp[%i] =%f - %f =%f\n", idx + 1, idx, tex1Dfetch(texRefx, (idx + 1)) , tex1Dfetch(texRefx, (idx)),tex1Dfetch(texRefx, (idx + 1))- tex1Dfetch(texRefx, (idx)));
		temp[threadIdx.x] = sqrtf(1 - powf((tex1Dfetch(texRefx, (idx)) + tex1Dfetch(texRefx, (idx + 1))) / 2.0, 2)) *(tex1Dfetch(texRefx, (idx + 1)) - tex1Dfetch(texRefx, (idx)));
		//printf("temp[%i] = %f\n", threadIdx.x, temp[threadIdx.x]);
		__syncthreads();
		if (threadIdx.x == 0)
		{
			float sum = 0.0;
			for (int j = 0; j < blockDim.x; j++)
			{
				//printf("CCCCtemp[%i]-temp[%i] =%f - %f =%f\n", j + 1, j, temp[j + 1], temp[j], temp[j + 1] - temp[j]);
				sum += temp[j];
				//printf("y[] = %f\n",  temp[j]);
			}
			atomicAdd(res, sum);
		}
	}
}

void SecondLab()
{
	int a = 0; int b = 1;
	float h = (b - a) / float(N);
	int memSize = sizeof(float);
	float *d_h;
	cudaMalloc((void **)&d_h, memSize);
	cudaMemcpy(d_h, &h, memSize, cudaMemcpyHostToDevice);

	float *h_x = (float*)malloc(N * memSize);
	for (int i = 0; i < N; i++)
	{
		h_x[i] = a + i*h;
		//printf("h_x[%i] = %f\n",i, h_x[i]);
	}

	float *d_x;
	cudaMalloc((void **)&d_x, N * memSize);

	eval_x << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_x, d_h);
	cudaBindTexture(0, texRefx, d_x, N * memSize);
	checkCUDAError("bind");

	float *d_y;
	cudaMalloc((void **)&d_y, memSize);
	float sum2 = 0;
	cudaMemcpy(d_y, &sum2, memSize, cudaMemcpyHostToDevice);
	//проверка на хосте
	printf("check int on host:\n");
	float *h_y = (float*)malloc(N * memSize);
	float sum = 0;
	for (int j = 0; j < N - 1; j++)
	{
		//printf("fun = %f\n", sqrtf(1 - powf((h_x[j] + h_x[j + 1]) / 2.0, 2)) * (h_x[j + 1] - h_x[j]));
		sum += sqrtf(1 - powf((h_x[j] + h_x[j + 1]) / 2.0, 2)) * (h_x[j + 1] - h_x[j]);
		//printf("h_x[%i] = %f\n",j, h_x[j]);
	}
	printf("int on host = %f:\n", sum * 4);
	eval_y_rect << <BLOCK_SIZE, THREADS_PER_BLOCK >> >(d_x, d_y);
	cudaMemcpy(&sum2, d_y, memSize, cudaMemcpyDeviceToHost);
	
	//for (int j = 0; j < N; j++)
	//{
	//	sum2 += h_y[j];
	//	//printf("h_y[%i] = %f\n",j, h_y[j]);
	//}
	printf("int on device = %f:\n", sum2 * 4);
	checkCUDAError("cudaMemcpy");
	cudaUnbindTexture(texRefx);
	checkCUDAError("cudaUnbindTexture");
	free(h_x); free(h_y);
	cudaFree(d_x); cudaFree(d_y); cudaFree(d_h);
}


texture<float, 1, cudaReadModeElementType> texRefx1;
texture<float, 1, cudaReadModeElementType> texRefy1;
texture<float, 1, cudaReadModeElementType> texRefx2;
texture<float, 1, cudaReadModeElementType> texRefy2;
texture<float, 1, cudaReadModeElementType> texRefx3;
texture<float, 1, cudaReadModeElementType> texRefy3;
__global__ void scalMult1(BASE_TYPE *C)
{
	__shared__ BASE_TYPE ash[THREADS_PER_BLOCK];
	ash[threadIdx.x] = tex1Dfetch(texRefx1, (blockIdx.x * blockDim.x + threadIdx.x)) * tex1Dfetch(texRefy1, (blockIdx.x * blockDim.x + threadIdx.x));
	__syncthreads();
	if (threadIdx.x == 0)
	{
		BASE_TYPE sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
		{
			sum += ash[j];
		}
		atomicAdd(C, sum);
	}
}

__global__ void scalMult2(BASE_TYPE *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ BASE_TYPE ash[THREADS_PER_BLOCK];
	//printf("tex1Dfetch(texRefx2, (blockIdx.x * blockDim.x + threadIdx.x)) = %f\n", tex1D(texRefx2, float(idx)) );
	//printf("tex1Dfetch(texRefy2, (blockIdx.x * blockDim.x + threadIdx.x)) = %f\n", tex1D(texRefy2, float(idx)) );
	ash[threadIdx.x] = tex1D(texRefx2, float(idx))*  tex1D(texRefy2, float(idx));
	__syncthreads();
	if (threadIdx.x == 0)
	{
		BASE_TYPE sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
		{
			sum += ash[j];
		}
		atomicAdd(C, sum);
	}
}
__global__ void scalMult3(BASE_TYPE *C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ BASE_TYPE ash[THREADS_PER_BLOCK];
	//printf("tex1Dfetch(texRefx2, (blockIdx.x * blockDim.x + threadIdx.x)) = %f\n", tex1D(texRefx2, float(idx)) );
	//printf("tex1Dfetch(texRefy2, (blockIdx.x * blockDim.x + threadIdx.x)) = %f\n", tex1D(texRefy2, float(idx)) );
	ash[threadIdx.x] = tex1Dfetch(texRefx3, (blockIdx.x * blockDim.x + threadIdx.x)) *  tex1D(texRefy3, float(idx));
	__syncthreads();
	if (threadIdx.x == 0)
	{
		BASE_TYPE sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
		{
			sum += ash[j];
		}
		atomicAdd(C, sum);
	}
}

void FirstHome()
{
	//оба массива связаны с лин. памятью
	float *h_res1 = (float*)malloc(sizeof(float));
	float *d_res1;
	cudaMalloc((void**)&d_res1, sizeof(float));
	float scRes1 = 0;
	cudaMemcpy(d_res1, &scRes1, sizeof(float), cudaMemcpyHostToDevice);

	float *d_x1;
	cudaMalloc((void**)&d_x1, sizeof(float) * N);
	cudaBindTexture(0, texRefx1, d_x1, sizeof(float) * N);

	float *d_y1;
	cudaMalloc((void**)&d_y1, sizeof(float) * N);
	cudaBindTexture(0, texRefy1, d_y1, sizeof(float) * N);

	float *h_x1 = (float*)malloc(N * sizeof(float));
	float *h_y1 = (float*)malloc(N * sizeof(float));
	for (int i = 0; i != N; i++)
	{
		h_x1[i] = rand() / (BASE_TYPE)RAND_MAX;
		h_y1[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	//check
	float check_res = 0;
	for (int i = 0; i != N; i++)
	{
		check_res += h_x1[i] * h_y1[i];
	}
	printf("check_res = %f\n", check_res);
	cudaMemcpy(d_x1, h_x1, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, h_y1, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaBindTexture(0, texRefx1, d_x1, sizeof(float) * N);
	cudaBindTexture(0, texRefy1, d_y1, sizeof(float) * N);
	// инициализируем события
	cudaEvent_t start, stop;
	float elapsedTime;
	// создаем события
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	scalMult1 << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_res1);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing by the GPU add with 2 lin mem: %.3f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaMemcpy(&scRes1, d_res1, sizeof(float), cudaMemcpyDeviceToHost);
	printf("add with 2 lin mem = %f\n", scRes1);
	cudaUnbindTexture(texRefx1);
	cudaUnbindTexture(texRefy1);

	//два cudaArr
	cudaArray* cuArrayX;
	cudaMallocArray(&cuArrayX, &texRefx2.channelDesc, N, 1);
	cudaMemcpyToArray(cuArrayX, 0, 0, h_x1, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texRefx2, cuArrayX);
	texRefx2.normalized = false;
	texRefx2.filterMode = cudaFilterModePoint;

	cudaArray* cuArrayY;
	cudaMallocArray(&cuArrayY, &texRefy2.channelDesc, N, 1);
	cudaMemcpyToArray(cuArrayY, 0, 0, h_y1, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texRefy2, cuArrayY);
	texRefy2.normalized = false;
	texRefy2.filterMode = cudaFilterModePoint;

	// обнуляем результат
	scRes1 = 0;
	cudaMemcpy(d_res1, &scRes1, sizeof(float), cudaMemcpyHostToDevice);

	// инициализируем события
	cudaEvent_t start2, stop2;
	float elapsedTime2;
	// создаем события
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	// запись события
	cudaEventRecord(start2, 0);
	// вызов ядра
	scalMult2 << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_res1);
	cudaEventRecord(stop2, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&elapsedTime2, start2, stop2);
	// вывод информации
	printf("Time spent executing by the GPU add with 2 cuda arr: %.3f millseconds\n", elapsedTime2);
	// уничтожение события
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);
	cudaMemcpy(&scRes1, d_res1, sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("add with 2 cuda arr = %f\n", scRes1);
	cudaUnbindTexture(texRefx2);
	cudaUnbindTexture(texRefy2);

	//смешанное
	cudaBindTexture(0, texRefx3, d_x1, sizeof(float) * N);

	cudaArray* cuArrayY1;
	cudaMallocArray(&cuArrayY1, &texRefy3.channelDesc, N, 1);
	cudaMemcpyToArray(cuArrayY1, 0, 0, h_y1, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texRefy3, cuArrayY1);
	texRefy3.normalized = false;
	texRefy3.filterMode = cudaFilterModePoint;

	// обнуляем результат
	scRes1 = 0;
	cudaMemcpy(d_res1, &scRes1, sizeof(float), cudaMemcpyHostToDevice);

	// инициализируем события
	cudaEvent_t start3, stop3;
	float elapsedTime3;
	// создаем события
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	// запись события
	cudaEventRecord(start3, 0);
	// вызов ядра
	scalMult3 << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_res1);
	cudaEventRecord(stop3, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&elapsedTime3, start3, stop3);
	// вывод информации
	printf("Time spent executing by the GPU add with 2 cuda arr: %.3f millseconds\n", elapsedTime3);
	// уничтожение события
	cudaEventDestroy(start3);
	cudaEventDestroy(stop3);
	cudaMemcpy(&scRes1, d_res1, sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("add with lin mem and cuda arr = %f\n", scRes1);
	cudaUnbindTexture(texRefx3);
	cudaUnbindTexture(texRefy3);
}

int main()
{
	//FirstLab();
	//SecondLab();
	FirstHome();
	cudaDeviceSynchronize();
	getchar();
	return 0;
}
