#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
//include <random>
using namespace std;

#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE double
// функция перемножения матриц
__global__ void matrixMult(const BASE_TYPE *A, BASE_TYPE *C, int Acols)
{
	int i0 = Acols * (blockDim.y * blockIdx.y + threadIdx.y);
	BASE_TYPE sum = 0;
	for (int k = 0; k < Acols; k++)
		sum += A[i0 + k] * A[i0 + k];
	int ind = Acols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	C[ind] = sum;
}


void FirstLab()
{
	// количество строк и столбцов матрицы int Arows = 100;
	int Acols = 5;
	int Arows = 5;
	//Arows = toMultiple(Arows, BLOCK_SIZE);
	printf("Arows = %d\n", Arows);
	//Acols = toMultiple(Acols, BLOCK_SIZE);
	printf("Acols = %d\n", Acols);
	size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
	size_t Csize = Arows * Acols * sizeof(BASE_TYPE);
	BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
	BASE_TYPE *h_C = (BASE_TYPE *)malloc(Csize);
	//инициализация матрицы A
	for (int i = 0; i < Arows * Acols; ++i) {
		h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	printf("A matrix:\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			printf("%.1f ", h_A[i*Acols + j]);
		}
		printf("\n");
	}

	BASE_TYPE *d_A = NULL;
	cudaMalloc((void **)&d_A, Asize);
	BASE_TYPE * d_C = NULL;
	cudaMalloc((void **)&d_C, Csize);

	cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(Acols / BLOCK_SIZE, Arows / BLOCK_SIZE);

	// Умножение
	matrixMult << <blocksPerGrid, threadsPerBlock >> >(d_A, d_C, Acols);

	cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);

	BASE_TYPE *h_E = (BASE_TYPE *)malloc(Asize);
	printf("E matrix:\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			if (i == j)
			{
				h_E[i*Acols + j] = 1;
			}
			else h_E[i*Acols + j] = 0;
			printf("%.1f  ", h_E[i*Acols + j]);
		}
		printf("\n");
	}

	printf("Test E-A*AT=0 ?:\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			h_E[i*Acols + j] -= h_C[i*Acols + j];
			printf("%.1f  ", h_E[i*Acols + j]);
		}
		printf("\n");
	}
	printf("Test PASSED\n");
	cudaFree(d_A);
	cudaFree(d_C);
	free(h_A);
}

void FirstHome()
{
	// количество строк и столбцов матрицы int Arows = 100;
	int Acols = 5;
	int Arows = 5;
	int Brows = 5;
	int Bcols = 5;
	//Arows = toMultiple(Arows, BLOCK_SIZE);
	printf("Arows = %d\n", Arows);
	//Acols = toMultiple(Acols, BLOCK_SIZE);
	printf("Acols = %d\n", Acols);
	//Brows = toMultiple(Brows, BLOCK_SIZE);
	printf("Brows = %d\n", Brows);
	//Bcols = toMultiple(Bcols, BLOCK_SIZE);
	printf("Bcols = %d\n", Bcols);
	size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
	size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
	size_t C1size = Arows * Bcols * sizeof(BASE_TYPE);
	size_t C2size = Brows * Acols * sizeof(BASE_TYPE);
	BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
	BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
	BASE_TYPE *h_C1 = (BASE_TYPE *)malloc(C1size);
	BASE_TYPE *h_C2 = (BASE_TYPE *)malloc(C2size);
	//инициализация матрицы A
	for (int i = 0; i < Arows * Acols; ++i) {
		h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	for (int i = 0; i < Arows * Acols; ++i) {
		h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
	}
	printf("A matrix:\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			printf("%f  ", h_A[i*Acols + j]);
		}
		printf("\n");
	}
	printf("B matrix:\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			printf("%f  ", h_B[i*Acols + j]);
		}
		printf("\n");
	}

	BASE_TYPE *d_A = NULL;
	cudaMalloc((void **)&d_A, Asize);
	BASE_TYPE *d_B = NULL;
	cudaMalloc((void **)&d_B, Bsize);
	BASE_TYPE * d_C1 = NULL;
	cudaMalloc((void **)&d_C1, C1size);
	BASE_TYPE * d_C2 = NULL;
	cudaMalloc((void **)&d_C2, C2size);

	cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(Acols / BLOCK_SIZE, Arows / BLOCK_SIZE);


	// Умножение A*B
	matrixMult << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C1, Acols, Bcols);
	cudaMemcpy(h_C1, d_C1, C1size, cudaMemcpyDeviceToHost);
	// Умножение B*A
	matrixMult << <blocksPerGrid, threadsPerBlock >> >(d_B, d_A, d_C2, Bcols, Acols);
	cudaMemcpy(h_C2, d_C2, C2size, cudaMemcpyDeviceToHost);

	double res;
	printf("Test C1-C2=0 ?:\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			res = h_C1[i*Acols + j] - h_C2[i*Acols + j];
			printf("%f  ", res);
		}
		printf("\n");
	}
	printf("Test PASSED\n");
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C1);
	cudaFree(d_C2);
	free(h_A);
	free(h_B);
	free(h_C1);
	free(h_C2);
}

__global__ void matrixAdd(BASE_TYPE **A, BASE_TYPE **B, BASE_TYPE **C, int cols)
{
	// Вычисление индекса элемента матрицы на GPU
	int i = (blockDim.y * blockIdx.y + threadIdx.y);
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("A[%i][%i] = %.1f\n", i, j, A[i][j]);
	//printf("B[%i][%i] = %.1f\n", i, j, B[i][j]);
	C[i][j] = A[i][j] + B[i][j];
	//printf("C[%i][%i] = %.1f\n", i, j, C[i][j]);
}

__global__ void show(BASE_TYPE **A, int cols)
{
	printf("Matrix on GPU:\n");
	for (int i = 0; i != cols; i++)
	{
		for (int j = 0; j != cols; j++)
		{
			printf("%.1f  ", A[i][j]);
		}
		printf("\n");
	}
}

void SecondHome()
{
	// количество строк и столбцов матрицы int Arows = 100;
	int Acols = 5;
	int Arows = 5;
	int Brows = 5;
	int Bcols = 5;
	Arows = toMultiple(Arows, BLOCK_SIZE);
	printf("Arows = %d\n", Arows);
	Acols = toMultiple(Acols, BLOCK_SIZE);
	printf("Acols = %d\n", Acols);
	Brows = toMultiple(Brows, BLOCK_SIZE);
	printf("Brows = %d\n", Brows);
	Bcols = toMultiple(Bcols, BLOCK_SIZE);
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
	BASE_TYPE **h_C = (BASE_TYPE **)malloc(Brows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Arows; i++) {
		h_C[i] = (BASE_TYPE*)malloc(Bcols * sizeof(BASE_TYPE));
	}
	//____________________________________________________________________

	//инициализация матрицы A,B,C
	for (int i = 0; i < Arows; ++i) {
		for (int j = 0; j<Acols; j++)
			h_A[i][j] = rand() / (BASE_TYPE)RAND_MAX;
	}
	for (int i = 0; i < Brows; ++i) {
		for (int j = 0; j<Bcols; j++)
			h_B[i][j] = rand() / (BASE_TYPE)RAND_MAX;
	}
	for (int i = 0; i < Brows; ++i) {
		for (int j = 0; j<Bcols; j++)
			h_C[i][j] = 0;
	}
	printf("A matrix:\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			printf("%.1f  ", h_A[i][j]);
		}
		printf("\n");
	}
	printf("B matrix:\n");
	for (int i = 0; i != Arows; i++)
	{
		for (int j = 0; j != Acols; j++)
		{
			printf("%.1f  ", h_B[i][j]);
		}
		printf("\n");
	}

	BASE_TYPE **d_A = NULL;
	cudaMalloc((void **)&d_A, Arows * sizeof(BASE_TYPE*)); // выделяем память на вектор из указателей на float
	BASE_TYPE **h_tempA = (BASE_TYPE **)malloc(Arows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Arows; i++) {
		cudaMalloc((void**)&h_tempA[i], Acols * sizeof(BASE_TYPE));
	}
	for (int i = 0; i<Arows; i++) {
		cudaMemcpy(h_tempA[i], h_A[i], Acols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_A, h_tempA, Arows * sizeof(BASE_TYPE*), cudaMemcpyHostToDevice);
	printf("A matrix:\n");
	show << <1, 1 >> >(d_A, Acols);

	BASE_TYPE **d_B = NULL;
	cudaMalloc((void **)&d_B, Brows * sizeof(BASE_TYPE*)); // выделяем память на вектор из указателей на float
	BASE_TYPE **h_tempB = (BASE_TYPE **)malloc(Brows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Brows; i++) {
		cudaMalloc((void**)&h_tempB[i], Bcols * sizeof(BASE_TYPE));
	}
	for (int i = 0; i<Arows; i++) {
		cudaMemcpy(h_tempB[i], h_B[i], Bcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_B, h_tempB, Brows * sizeof(BASE_TYPE*), cudaMemcpyHostToDevice);
	printf("B matrix:\n");
	show << <1, 1 >> >(d_B, Bcols);

	BASE_TYPE **d_C = NULL;
	cudaMalloc((void **)&d_C, Brows * sizeof(BASE_TYPE*)); // выделяем память на вектор из указателей на float
	BASE_TYPE **h_tempC = (BASE_TYPE **)malloc(Brows * sizeof(BASE_TYPE*));
	for (int i = 0; i<Brows; i++) {
		cudaMalloc((void**)&h_tempC[i], Bcols * sizeof(BASE_TYPE));
	}
	for (int i = 0; i<Arows; i++) {
		cudaMemcpy(h_tempC[i], h_C[i], Bcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_C, h_tempC, Brows * sizeof(BASE_TYPE*), cudaMemcpyHostToDevice);
	//show << <1, 1 >> >(d_C, Bcols);

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(Acols / BLOCK_SIZE, Arows / BLOCK_SIZE);
	matrixAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, Acols);
	printf("Result\n");
	show << <1, 1 >> >(d_C, Bcols);
	free(h_A);
	free(h_B);
	free(h_C);
}

int main()
{
	FirstLab();

	cudaDeviceSynchronize();
	getchar();
	return 0;
}
