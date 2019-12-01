// стр 45 лаб 1
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
using namespace std;
#define MAX_GRIDSIZE 1
#define N 100
#define BLOCKS 18
#define threads_per_block 128
#define PI 3.141592653
#define BASE_TYPE float

#pragma region CUDA KERNELS


__global__ void eval_cosf(float* x)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		printf("idx == %i\n---------------------------\n", idx);
		x[idx] = sinf((idx % 360)*PI / 360);
		printf("d_x[%i] = %f\n", idx, x[idx]);
	}
}

__global__ void eval_expf(float* x)
{
	int idx = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		printf("idx == %i\n---------------------------\n", idx);
		x[idx] = __expf(idx);
		printf("d_x[%i] = %f\n", idx, x[idx]);
	}
}

__global__ void eval_logf(float* x)
{
	int idx = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		printf("idx == %i\n---------------------------\n", idx);
		x[idx] = logf(idx + 1);
		printf("d_x[%i] = %f\n", idx, x[idx]);
	}
}

__global__ void eval_tanf(float* x)
{
	int idx = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		x[idx] = tanf(idx);
		//printf("d_x[%i] = %f\n", idx, x[idx]);
	}
}

__global__ void eval_powf(float* x)
{
	int idx = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		x[idx] = powf(idx, 2);
	}
}

__global__ void eval_sqrtf(float* x)
{
	int idx = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		x[idx] = sqrtf(idx);
	}
}


__global__ void fadd(float * x, float *y, float *res)
{
	int index = (blockIdx.y * MAX_GRIDSIZE + blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < N)
	{
		res[index] = __fadd_ru(x[index], y[index]);
	}
}

__global__ void dadd(double * x, double *y, double *res)
{
	int index = (blockIdx.y * MAX_GRIDSIZE + blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < N)
	{
		res[index] = __dadd_ru(x[index], y[index]);
	}
}

__global__ void fsub(float * x, float *y, float *res)
{
	int index = (blockIdx.y * MAX_GRIDSIZE + blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < N)
	{
		res[index] = __fsub_ru(x[index], y[index]);
	}
}

__global__ void dsub(double * x, double *y, double *res)
{
	int index = (blockIdx.y * MAX_GRIDSIZE + blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < N)
	{
		res[index] = __dsub_ru(x[index], y[index]);
	}
}



__global__ void eval_cosd(double* x)
{
	int idx = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		x[idx] = cos((idx % 360)*PI / 360);
		//printf("d_x[%i] = %f\n", idx, x[idx]);
	}
}

__global__ void eval_sinf(float* x)
{
	int idx = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		x[idx] = sinf((idx % 360)*PI / 360);
		//printf("d_x[%i] = %f\n", idx, x[idx]);
	}
}

__global__ void eval_sind(double* x)
{
	int idx = (blockIdx.y*MAX_GRIDSIZE + blockIdx.x)*blockDim.x + threadIdx.x;
	if (idx < N)
	{
		x[idx] = sin((idx % 360)*PI / 360);
		//printf("d_x[%i] = %f\n", idx, x[idx]);
	}
}


__global__ void show(BASE_TYPE **A, int cols)
{
	printf("Matrix on GPU:\n");
	for (int i = 0; i != cols; i++)
	{
		for (int j = 0; j != cols; j++)
		{
			printf("%f  ", A[i][j]);
		}
		printf("\n");
	}
}


__global__ void vectorAdd(BASE_TYPE *a, BASE_TYPE *b, int Acols)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	//проверка на выход за пределы массива
	if (tid < Acols)
	{
		//printf("a[%i] = %f\n", tid, a[tid]);
		//printf("b[%i] = %f\n", tid, b[tid]);
		a[tid] += b[tid];
		//printf("aftr add a[%i] = %f\n", tid, a[tid]);
	}
	else
		return;
}
__global__ void vectorSub(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *c, int Acols)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	//проверка на выход за пределы массива
	if (tid < Acols)
		c[tid] = a[tid] - b[tid];
	else
		return;
}

__global__ void scalMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C)
{
	BASE_TYPE sum = 0.0;
	__shared__ BASE_TYPE ash[threads_per_block];
	__shared__ BASE_TYPE bsh[threads_per_block];
	ash[threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x];
	bsh[threadIdx.x] = B[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
		{
			//printf("a[%i] = %f\n", j, ash[j]);
			//printf("b[%i] = %f\n", j, bsh[j]);
			sum += ash[j] * bsh[j];
		}
		C[blockIdx.x] = sum;
		//printf("C = %f\n", C[blockIdx.x]);
	}
}

__global__ void scalMult(const BASE_TYPE *A, BASE_TYPE *C)
{
	BASE_TYPE sum = 0.0;
	__shared__ BASE_TYPE ash[threads_per_block];
	__shared__ BASE_TYPE bsh[threads_per_block];
	ash[threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] * A[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		sum = 0.0;
		for (int j = 0; j < blockDim.x; j++)
		{
			//printf("a[%i] = %f\n", j, ash[j]);
			sum += ash[j];
		}
		C[blockIdx.x] = sum;
		//printf("C = %f\n", C[blockIdx.x]);
	}
}

__global__ void scalOnVector(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int Acols)
{
	//printf("AB = %f\n",*A);
	//printf("BB = %f\n", *B);
	BASE_TYPE frac = 1.0*(*A) / (*B);
	//printf("frac = %f\n", frac);
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < Acols)
	{
		C[index] *= frac;
		//printf("C[%i] = %f\n", index, C[index]);
	}
}

__global__ void difVectors(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int Acols)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index<Acols)
	{
		C[index] = A[index] - B[index];
	}
}
#pragma endregion

#pragma region HOST METHODS


void GetCosError()
{
	float * d_x;
	cudaMalloc((void**)&d_x, N * sizeof(float));
	eval_cosf << <BLOCKS, threads_per_block >> > (d_x);

	//копирование с gpu на cpu
	float *dh_x;
	dh_x = (float*)malloc(N * sizeof(float));
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

	float err = 0;
	for (int i = 0; i != N; i++)
	{
		//printf("dh_x[%i] = %f\n", i, dh_x[i]);
		err += abs(cos((i % 360)*PI / 180) - dh_x[i]);
	}
	err /= N;
	printf("cos err= %f\n", err);
	free(dh_x);
	cudaFree(d_x);
}

void GetExpError()
{
	float * d_x;
	cudaMalloc((void**)&d_x, N * sizeof(float));
	eval_cosf << <BLOCKS, threads_per_block >> > (d_x);
	//копирование с gpu на cpu
	float *dh_x;
	dh_x = (float*)malloc(N * sizeof(float));
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

	float err = 0;

	eval_expf << <BLOCKS, threads_per_block >> > (d_x);
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i != N; i++)
	{
		//printf("dh_x[%i] = %f\n", i, dh_x[i]);
		err += abs(exp(i) - dh_x[i]);
	}
	err /= N;
	printf("exp err= %f\n", err);

	free(dh_x);
	cudaFree(d_x);
}

void GetLogError()
{
	float * d_x;
	cudaMalloc((void**)&d_x, N * sizeof(float));
	eval_cosf << <BLOCKS, threads_per_block >> > (d_x);
	//копирование с gpu на cpu
	float *dh_x;
	dh_x = (float*)malloc(N * sizeof(float));
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
	float err = 0;
	eval_logf << <BLOCKS, threads_per_block >> > (d_x);
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 1; i != N; i++)
	{
		//printf("dh_x[%i] = %f\n", i, dh_x[i]);
		err += abs(log(i) - dh_x[i]);
		/*printf("log(%i) = %f\n", i,log(i));
		printf("dh_x[%i] = %f\n", i , dh_x[i]);*/
	}
	err /= N;
	printf("log err= %f\n", err);

	free(dh_x);
	cudaFree(d_x);
}

void GetTanError()
{
	float * d_x;
	cudaMalloc((void**)&d_x, N * sizeof(float));
	eval_cosf << <BLOCKS, threads_per_block >> > (d_x);
	//копирование с gpu на cpu
	float *dh_x;
	dh_x = (float*)malloc(N * sizeof(float));
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
	float err = 0;	
	eval_tanf << <BLOCKS, threads_per_block >> > (d_x);
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i != N; i++)
	{
		//printf("dh_x[%i] = %f\n", i, dh_x[i]);
		err += abs(tan(i) - dh_x[i]);
		printf("tan(%i) = %f\n", i, tan(i));
		printf("dh_x[%i] = %f\n", i, dh_x[i]);
	}
	err /= N;
	printf("tan err= %.0e\n", err);

	free(dh_x);
	cudaFree(d_x);
}

void GetPowError()
{
	float * d_x;
	cudaMalloc((void**)&d_x, N * sizeof(float));
	eval_cosf << <BLOCKS, threads_per_block >> > (d_x);
	//копирование с gpu на cpu
	float *dh_x;
	dh_x = (float*)malloc(N * sizeof(float));
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
	float err = 0;
	eval_powf << <BLOCKS, threads_per_block >> > (d_x);
	cudaMemcpy(dh_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i != N; i++)
	{
		//printf("dh_x[%i] = %f\n", i, dh_x[i]);
		err += abs(powf(i, 2) - dh_x[i]);
		/*printf("tan(%i) = %f\n", i, tan(i));
		printf("dh_x[%i] = %f\n", i, dh_x[i]);*/
	}
	err /= N;
	printf("pow err= %f\n", err);

	free(dh_x);
	cudaFree(d_x);
}

void FirstLabTask()
{
	GetCosError();
	GetExpError();
	GetLogError();
	GetTanError();
	GetPowError();
}

void CheckTimeAndErrorAdd()
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float *fx;
	fx = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		fx[i] = rand() / (float)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	float *fy;
	fy = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		fy[i] = rand() / (float)RAND_MAX;
		//printf("y[%i] = %f\n", i, y[i]);
	}
	float *fres;
	fres = (float*)malloc(sizeof(float)*N);
	float * fd_x;
	cudaMalloc((void**)&fd_x, N * sizeof(float));
	cudaMemcpy(fd_x, fx, N * sizeof(float), cudaMemcpyHostToDevice);
	float * fd_y;
	cudaMalloc((void**)&fd_y, N * sizeof(float));
	cudaMemcpy(fd_y, fy, N * sizeof(float), cudaMemcpyHostToDevice);
	float * fd_res;
	cudaMalloc((void**)&fd_res, N * sizeof(float));
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	fadd << <BLOCKS, threads_per_block >> > (fd_x, fd_y, fd_res);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing __fadd_ru by the GPU: %f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime = 0.0;
	cudaMemcpy(fres, fd_res, N * sizeof(float), cudaMemcpyDeviceToHost);

	//dadd
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double *dx;
	dx = (double*)malloc(sizeof(double)*N);
	for (int i = 0; i < N; ++i) {
		dx[i] = rand() / (double)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	double *dy;
	dy = (double*)malloc(sizeof(double)*N);
	for (int i = 0; i < N; ++i) {
		dy[i] = rand() / (double)RAND_MAX;
		//printf("y[%i] = %f\n", i, y[i]);
	}
	double *dres;
	dres = (double*)malloc(sizeof(double)*N);
	double * dd_x;
	cudaMalloc((void**)&dd_x, N * sizeof(double));
	cudaMemcpy(dd_x, dx, N * sizeof(double), cudaMemcpyHostToDevice);
	double * dd_y;
	cudaMalloc((void**)&dd_y, N * sizeof(double));
	cudaMemcpy(dd_y, dy, N * sizeof(double), cudaMemcpyHostToDevice);
	double * dd_res;
	cudaMalloc((void**)&dd_res, N * sizeof(double));
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	dadd << <BLOCKS, threads_per_block >> > (dd_x, dd_y, dd_res);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing __dadd_ru by the GPU: %f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime = 0.0;
	cudaMemcpy(dres, dd_res, N * sizeof(double), cudaMemcpyDeviceToHost);

	//подсчет ошибки
	float errf = 0.0;
	double errd = 0.0;
	for (int i = 0; i != N; i++)
	{
		errf += abs((fx[i] + fy[i]) - fres[i]);
		errd += abs((dx[i] + dy[i]) - dres[i]);
	}
	printf("fadd error = %f\n", errf);
	printf("dadd error = %f\n", errd);
	errf = 0.0;
	errd = 0.0;

	cudaFree(fd_x); cudaFree(fd_y); cudaFree(fd_res);
}

void CheckTimeAndErrorSub()
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float *fx;
	fx = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		fx[i] = rand() / (float)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	float *fy;
	fy = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		fy[i] = rand() / (float)RAND_MAX;
		//printf("y[%i] = %f\n", i, y[i]);
	}
	float *fres;
	fres = (float*)malloc(sizeof(float)*N);
	float * fd_x;
	cudaMalloc((void**)&fd_x, N * sizeof(float));
	float * fd_y;
	cudaMalloc((void**)&fd_y, N * sizeof(float));
	float * fd_res;
	cudaMalloc((void**)&fd_res, N * sizeof(float));

	cudaMemcpy(fd_x, fx, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fd_y, fy, N * sizeof(float), cudaMemcpyHostToDevice);
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	fsub << <BLOCKS, threads_per_block >> > (fd_x, fd_y, fd_res);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing __fsub_ru by the GPU: %f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime = 0.0;
	cudaMemcpy(fres, fd_res, N * sizeof(float), cudaMemcpyDeviceToHost);

	//dsub
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double *dx;
	dx = (double*)malloc(sizeof(double)*N);
	for (int i = 0; i < N; ++i) {
		dx[i] = rand() / (double)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	double *dy;
	dy = (double*)malloc(sizeof(double)*N);
	for (int i = 0; i < N; ++i) {
		dy[i] = rand() / (double)RAND_MAX;
		//printf("y[%i] = %f\n", i, y[i]);
	}

	double *dres;
	dres = (double*)malloc(sizeof(double)*N);
	double * dd_x;
	cudaMalloc((void**)&dd_x, N * sizeof(double));
	cudaMemcpy(dd_x, dx, N * sizeof(double), cudaMemcpyHostToDevice);
	double * dd_y;
	cudaMalloc((void**)&dd_y, N * sizeof(double));
	cudaMemcpy(dd_y, dy, N * sizeof(double), cudaMemcpyHostToDevice);
	double * dd_res;
	cudaMalloc((void**)&dd_res, N * sizeof(double));

	cudaMemcpy(dd_x, dx, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dd_y, dy, N * sizeof(double), cudaMemcpyHostToDevice);
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	dsub << <BLOCKS, threads_per_block >> > (dd_x, dd_y, dd_res);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing __dsub_ru by the GPU: %f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime = 0.0;
	cudaMemcpy(dres, dd_res, N * sizeof(double), cudaMemcpyDeviceToHost);

	//подсчет ошибки
	float errf = 0.0;
	double errd = 0.0;
	for (int i = 0; i != N; i++)
	{
		errf += abs((fx[i] - fy[i]) - fres[i]);
		errd += abs((dx[i] - dy[i]) - dres[i]);
	}
	printf("fsub error = %f\n", errf);
	printf("dsub error = %f\n", errd);
	cudaFree(fd_x); cudaFree(fd_y); cudaFree(fd_res);
}

void CheckTimeAndErrorCos()
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float *fx;
	fx = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		fx[i] = rand() / (float)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	float *fy;
	fy = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		fy[i] = rand() / (float)RAND_MAX;
		//printf("y[%i] = %f\n", i, y[i]);
	}
	float *fres;
	fres = (float*)malloc(sizeof(float)*N);
	float * fd_x;
	cudaMalloc((void**)&fd_x, N * sizeof(float));
	float * fd_y;
	cudaMalloc((void**)&fd_y, N * sizeof(float));
	float * fd_res;
	cudaMalloc((void**)&fd_res, N * sizeof(float));

	cudaMemcpy(fd_x, fx, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fd_y, fy, N * sizeof(float), cudaMemcpyHostToDevice);
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	eval_cosf << <BLOCKS, threads_per_block >> > (fd_x);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing __cosf by the GPU: %f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime = 0.0;
	cudaMemcpy(fres, fd_res, N * sizeof(float), cudaMemcpyDeviceToHost);

	//dsub
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double *dx;
	dx = (double*)malloc(sizeof(double)*N);
	for (int i = 0; i < N; ++i) {
		dx[i] = rand() / (double)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	double *dy;
	dy = (double*)malloc(sizeof(double)*N);
	for (int i = 0; i < N; ++i) {
		dy[i] = rand() / (double)RAND_MAX;
		//printf("y[%i] = %f\n", i, y[i]);
	}

	double *dres;
	dres = (double*)malloc(sizeof(double)*N);
	double * dd_x;
	cudaMalloc((void**)&dd_x, N * sizeof(double));
	cudaMemcpy(dd_x, dx, N * sizeof(double), cudaMemcpyHostToDevice);
	double * dd_y;
	cudaMalloc((void**)&dd_y, N * sizeof(double));
	cudaMemcpy(dd_y, dy, N * sizeof(double), cudaMemcpyHostToDevice);
	double * dd_res;
	cudaMalloc((void**)&dd_res, N * sizeof(double));

	cudaMemcpy(dd_x, dx, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dd_y, dy, N * sizeof(double), cudaMemcpyHostToDevice);
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	eval_cosd << <BLOCKS, threads_per_block >> > (dd_x);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing eval_cosd by the GPU: %f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime = 0.0;
	cudaMemcpy(dres, dd_res, N * sizeof(double), cudaMemcpyDeviceToHost);

	//подсчет ошибки
	float errf = 0.0;
	double errd = 0.0;
	for (int i = 0; i != N; i++)
	{
		errf += abs(cosf((i % 360)*PI / 180) - fres[i]);
		errd += abs(cos((i % 360)*PI / 180) - dres[i]);
	}
	errf /= N;
	errd /= N;
	printf("fcos error = %f\n", errf);
	printf("dcos error = %f\n", errd);
	cudaFree(fd_x); cudaFree(fd_y); cudaFree(fd_res);
}

void CheckTimeAndErrorSin()
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float *fx;
	fx = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		fx[i] = rand() / (float)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	float *fy;
	fy = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; ++i) {
		fy[i] = rand() / (float)RAND_MAX;
		//printf("y[%i] = %f\n", i, y[i]);
	}
	float *fres;
	fres = (float*)malloc(sizeof(float)*N);
	float * fd_x;
	cudaMalloc((void**)&fd_x, N * sizeof(float));
	float * fd_y;
	cudaMalloc((void**)&fd_y, N * sizeof(float));
	float * fd_res;
	cudaMalloc((void**)&fd_res, N * sizeof(float));

	cudaMemcpy(fd_x, fx, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fd_y, fy, N * sizeof(float), cudaMemcpyHostToDevice);
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	eval_sinf << <BLOCKS, threads_per_block >> > (fd_x);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing eval_sinf by the GPU: %f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime = 0.0;
	cudaMemcpy(fres, fd_res, N * sizeof(float), cudaMemcpyDeviceToHost);

	//dsub
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double *dx;
	dx = (double*)malloc(sizeof(double)*N);
	for (int i = 0; i < N; ++i) {
		dx[i] = rand() / (double)RAND_MAX;
		//printf("x[%i] = %f\n", i, x[i]);
	}
	double *dy;
	dy = (double*)malloc(sizeof(double)*N);
	for (int i = 0; i < N; ++i) {
		dy[i] = rand() / (double)RAND_MAX;
		//printf("y[%i] = %f\n", i, y[i]);
	}

	double *dres;
	dres = (double*)malloc(sizeof(double)*N);
	double * dd_x;
	cudaMalloc((void**)&dd_x, N * sizeof(double));
	cudaMemcpy(dd_x, dx, N * sizeof(double), cudaMemcpyHostToDevice);
	double * dd_y;
	cudaMalloc((void**)&dd_y, N * sizeof(double));
	cudaMemcpy(dd_y, dy, N * sizeof(double), cudaMemcpyHostToDevice);
	double * dd_res;
	cudaMalloc((void**)&dd_res, N * sizeof(double));

	cudaMemcpy(dd_x, dx, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dd_y, dy, N * sizeof(double), cudaMemcpyHostToDevice);
	// запись события
	cudaEventRecord(start, 0);
	// вызов ядра
	eval_sind << <BLOCKS, threads_per_block >> > (dd_x);
	cudaEventRecord(stop, 0);
	// ожидание завершения работы ядра
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// вывод информации
	printf("Time spent executing eval_sind by the GPU: %f millseconds\n", elapsedTime);
	// уничтожение события
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	elapsedTime = 0.0;
	cudaMemcpy(dres, dd_res, N * sizeof(double), cudaMemcpyDeviceToHost);

	//подсчет ошибки
	float errf = 0.0;
	double errd = 0.0;
	for (int i = 0; i != N; i++)
	{
		errf += abs(cosf((i % 360)*PI / 180) - fres[i]);
		errd += abs(cos((i % 360)*PI / 180) - dres[i]);
	}
	errf /= N;
	errd /= N;
	printf("fsin error = %f\n", errf);
	printf("dsin error = %f\n", errd);
	cudaFree(fd_x); cudaFree(fd_y); cudaFree(fd_res);
}

void FirstHomeTask()
{
	CheckTimeAndErrorAdd();
	CheckTimeAndErrorSub();
	CheckTimeAndErrorCos();
	CheckTimeAndErrorSin();
}

int toMultiple(int a, int b) 
{
	int mod = a % b;
	if (mod != 0) {
		mod = b - mod;
		return a + mod;
	}
	return a;
}

void SecondHomeTask()
{
	int Acols = 3;
	int Arows = 3;
	int Brows = 3;
	int Bcols = 3;
	Arows = toMultiple(Arows, BLOCKS);
	printf("Arows = %d\n", Arows);
	Acols = toMultiple(Acols, BLOCKS);
	printf("Acols = %d\n", Acols);
	Brows = toMultiple(Brows, BLOCKS);
	printf("Brows = %d\n", Brows);
	Bcols = toMultiple(Bcols, BLOCKS);
	printf("Bcols = %d\n", Bcols);
	size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
	size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
	size_t Csize = Arows * Acols * sizeof(BASE_TYPE);
	//___________________VECTOR_OF_VECTORS________________________________
	BASE_TYPE **h_A = (BASE_TYPE **)malloc(Arows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Arows; i++) {
		h_A[i] = (BASE_TYPE*)malloc(Acols * sizeof(BASE_TYPE));
	}
	BASE_TYPE **h_B = (BASE_TYPE **)malloc(Brows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Arows; i++) {
		h_B[i] = (BASE_TYPE*)malloc(Bcols * sizeof(BASE_TYPE));
	}
	//____________________________________________________________________

	for (int i = 0; i < Arows; ++i) {
		for (int j = 0; j < Acols; j++)
		{
			if (j >= i)
			{
				h_A[i][j] = 1;
			}
			else
			{
				h_A[i][j] = 0;
			}
		}
	}
	for (int i = 0; i < Brows; ++i) {
		for (int j = 0; j < Bcols; j++)
			h_B[i][j] = 0;
	}



	///check
	BASE_TYPE **h_A1 = (BASE_TYPE **)malloc(Arows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Arows; i++) {
		h_A1[i] = (BASE_TYPE*)malloc(Acols * sizeof(BASE_TYPE));
	}
	BASE_TYPE **h_B1 = (BASE_TYPE **)malloc(Brows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Arows; i++) {
		h_B1[i] = (BASE_TYPE*)malloc(Bcols * sizeof(BASE_TYPE));
	}
	//____________________________________________________________________

	for (int i = 0; i < Arows; ++i) {
		for (int j = 0; j < Acols; j++)
		{
			if (j >= i)
			{
				h_A1[i][j] = 1;
			}
			else
			{
				h_A1[i][j] = 0;
			}
		}
	}
	for (int i = 0; i < Brows; ++i) {
		for (int j = 0; j < Bcols; j++)
			h_B1[i][j] = 0;
	}

	for (int i = 0; i != Acols; i++)
	{
		h_B1[0][i] = h_A1[0][i];
	}

	for (int i = 1; i != Arows; i++)
	{
		BASE_TYPE * temp = (BASE_TYPE*)malloc(Acols * sizeof(BASE_TYPE));
		for (int j = 0; j != Acols; j++)
		{
			temp[j] = 0;
		}
		double tempAB = 0;
		double tempBB = 0;
		for (int k = 0; k <= i - 1; k++)
		{
			tempAB = 0.0;
			tempBB = 0.0;

			for (int j = 0; j != Acols; j++)
			{
				//printf("h_A1[%i][%i] = %f\n",i,j, h_A1[i][j]);
				//printf("h_B1[%i][%i] = %f\n", k, j, h_B1[k][j]);
				tempAB += h_A1[i][j] * h_B1[k][j];
				tempBB += h_B1[k][j] * h_B1[k][j];
			}
			for (int j = 0; j != Acols; j++)
			{
				//printf("h_B1[%i][%i] = %f\n", k, j, h_B1[k][j]);
				temp[j] += h_B1[k][j] * tempAB*1.0 / tempBB;
				//printf("temp[%i] = %f\n", j, temp[j]);
			}
			//printf("scal ab = %f\n", tempAB);
			//printf("scal bb = %f\n", tempBB);
		}

		for (int j = 0; j != Acols; j++)
		{
			h_B1[i][j] = h_A1[i][j] - temp[j];
			//printf("h_B1[%i][%i] = %f\n", i, j, h_B1[i][j]);
		}
		free(temp);
	}
	double sumcheck = 0;
	/*for (int i = 1; i != Arows; i++)
	{*/
	for (int j = 0; j != Acols; j++)
		//sumcheck += h_B1[1][j] * h_B1[5][j];
		//printf("sumcheck = %f\n", sumcheck);
		sumcheck = 0;
	//}
	//cout << "sumcheck =" << sumcheck << endl;
	//printf("\n check\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			printf("%f  ", h_B1[i][j]);
		}
		printf("\n");
	}

	///


	BASE_TYPE **d_A = NULL;
	cudaMalloc((void **)&d_A, Arows * sizeof(BASE_TYPE*));
	BASE_TYPE **h_tempA = (BASE_TYPE **)malloc(Arows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Arows; i++) {
		cudaMalloc((void**)&h_tempA[i], Acols * sizeof(BASE_TYPE));
	}
	for (int i = 0; i<Arows; i++) {
		cudaMemcpy(h_tempA[i], h_A[i], Acols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_A, h_tempA, Arows * sizeof(BASE_TYPE*), cudaMemcpyHostToDevice);
	//printf("A matrix:\n");
	//show << <1, 1 >> >(d_A, Acols);

	BASE_TYPE **d_B = NULL;
	cudaMalloc((void **)&d_B, Brows * sizeof(BASE_TYPE*));
	BASE_TYPE **h_tempB = (BASE_TYPE **)malloc(Brows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Brows; i++) {
		cudaMalloc((void**)&h_tempB[i], Bcols * sizeof(BASE_TYPE));
	}
	cudaMemcpy(h_tempB[0], h_A[0], Bcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);//copy first vect
	for (int i = 1; i<Arows; i++) {
		cudaMemcpy(h_tempB[i], h_B[i], Bcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_B, h_tempB, Brows * sizeof(BASE_TYPE*), cudaMemcpyHostToDevice);
	//printf("B matrix:\n");
	//show << <1, 1 >> >(d_B, Bcols);

	BASE_TYPE *zeroVec;
	zeroVec = (BASE_TYPE*)malloc(Acols * sizeof(BASE_TYPE));
	for (int i = 0; i != Acols; i++)
	{
		zeroVec[i] = 0;
	}


	BASE_TYPE *scalAB;
	scalAB = (BASE_TYPE*)malloc(sizeof(BASE_TYPE));
	BASE_TYPE *scalBB;
	scalBB = (BASE_TYPE*)malloc(sizeof(BASE_TYPE));
	BASE_TYPE *scalABtemp;
	scalABtemp = (BASE_TYPE*)malloc(sizeof(BASE_TYPE));
	BASE_TYPE *scalBBtemp;
	scalBBtemp = (BASE_TYPE*)malloc(sizeof(BASE_TYPE));
	for (int i = 1; i != Brows; i++)
	{
		BASE_TYPE * d_tempSum;
		cudaMalloc((void**)&d_tempSum, Acols * sizeof(BASE_TYPE));
		cudaMemcpy(d_tempSum, zeroVec, Acols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);


		BASE_TYPE *zero;
		zero = (BASE_TYPE*)malloc(sizeof(BASE_TYPE));
		*zero = 0;
		for (int k = 0; k <= i - 1; k++)
		{
			BASE_TYPE * d_scalAB;
			cudaMalloc((void**)&d_scalAB, sizeof(BASE_TYPE));
			BASE_TYPE * d_scalBB;
			cudaMalloc((void**)&d_scalBB, sizeof(BASE_TYPE));

			BASE_TYPE * d_tempMul;
			cudaMalloc((void**)&d_tempMul, Acols * sizeof(BASE_TYPE));
			//cudaMemcpy(d_tempSum, zeroVec, Acols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
			///check h_tempB[k]
			BASE_TYPE *checkB = (BASE_TYPE*)malloc(Acols * sizeof(BASE_TYPE));
			cudaMemcpy(checkB, h_tempB[k], Acols * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
			//printf("k = %i\n", k);
			for (int n = 0; n != Acols; n++)
			{
				//printf("checkB[%i] = %f\n",n, checkB[n]);
			}
			///check h_tempA[i]
			BASE_TYPE *checkA = (BASE_TYPE*)malloc(Acols * sizeof(BASE_TYPE));
			cudaMemcpy(checkA, h_tempA[i], Acols * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
			//printf("i = %i\n", i);
			for (int n = 0; n != Acols; n++)
			{
				//printf("checkA[%i] = %f\n",n, checkA[n]);
			}
			///
			scalMult << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (h_tempA[i], h_tempB[k], d_scalAB);
			cudaMemcpy(scalAB, d_scalAB, sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
			//printf("scalAB = %f\n", *scalAB);

			scalMult << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (h_tempB[k], d_scalBB);
			cudaMemcpy(scalBB, d_scalBB, sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
			//printf("scalBB = %f\n", *scalBB);

			cudaMemcpy(d_tempMul, h_tempB[k], Acols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
			scalOnVector << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_scalAB, d_scalBB, d_tempMul, Acols);
			///check h_tempSum
			BASE_TYPE *checktempSum = (BASE_TYPE*)malloc(Acols * sizeof(BASE_TYPE));
			cudaMemcpy(checktempSum, d_tempSum, Acols * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
			//printf("i = %i\n", i);
			for (int n = 0; n != Acols; n++)
			{
				//printf("before sum checktempSum[%i] = %f\n", n, checktempSum[n]);
			}
			///
			vectorAdd << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (d_tempSum, d_tempMul, Acols);
			///
			cudaMemcpy(checktempSum, d_tempSum, Acols * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
			for (int n = 0; n != Acols; n++)
			{
				//printf("aftr sum checktempSum[%i] = %f\n", n, checktempSum[n]);
			}
			cudaFree(d_tempMul); cudaFree(d_scalAB); cudaFree(d_scalBB);
			///
		}
		vectorSub << <BLOCK_SIZE, THREADS_PER_BLOCK >> > (h_tempA[i], d_tempSum, h_tempB[i], Acols);
		//check h_tempB[k]
		BASE_TYPE *checkB = (BASE_TYPE*)malloc(Acols * sizeof(BASE_TYPE));
		cudaMemcpy(checkB, h_tempB[i], Acols * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
		//printf("i = %i\n", i);
		for (int n = 0; n != Acols; n++)
		{
			//printf("checkB[%i] = %f\n", n, checkB[n]);
		}
		//
		cudaFree(d_tempSum); //cudaFree(d_scalAB); cudaFree(d_scalBB);
	}
	printf("B matrix:\n");
	show << <1, 1 >> >(d_B, Bcols);
}
#pragma endregion



int main()
{
	//FirstLabTask();
	FirstHomeTask();

	cudaDeviceSynchronize();
	getchar();
	return 0;
}

