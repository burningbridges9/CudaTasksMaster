#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
// ����������� ���������� cuBLAS
#include <cublas_v2.h>
#include <cublasXt.h>


#define IDX2C(i,j,ld) (((i)*(ld))+(j))

__global__ void show(float **A, int cols, int rows)
{
	printf("Matrix on GPU:\n");
	for (int i = 0; i != rows; i++)
	{
		for (int j = 0; j != cols; j++)
		{
			printf("%.1f  ", A[i][j]);
		}
		printf("\n");
	}
}

void CublasExample()
{
	//int batch_size = 1;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	const int N = 6;
	float *dev_A, *dev_b, **dev_Aarray;
	float *x, *A, *b, **Aarray;
	/*int * info_array;
	int * dev_info_array;*/
	x = (float *)malloc(N * sizeof(*x));
	if (!x) {
		printf("host memory allocation failed");
		//return EXIT_FAILURE;
	}
	b = (float *)malloc(N * sizeof(*b));
	if (!b) {
		printf("host memory allocation failed");
		//return EXIT_FAILURE;
	}
	A = (float *)malloc(N * N * sizeof(*A));
	if (!A) {
		printf("host memory allocation failed");
		//return EXIT_FAILURE;
	}
	/*info_array = (int *)malloc(batch_size * sizeof(*info_array));
	Aarray = (float **)malloc(sizeof(float*));
	Aarray[0] = (float *)malloc(N * N * sizeof(*A));*/
	// ������������� ������� � ������� ������ �����
	int ind = 11;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			if (i >= j)
				A[IDX2C(i, j, N)] = (float)ind++;
			else A[IDX2C(i, j, N)] = 0.0f;
			b[i] = 1.0f;
	}
#pragma region  HomeFirst

	//Aarray[0] = A;
	//printf("Generated matrix A:\n");
	//for (int k = 0; k < 1; k++)
	//{
	//	for (int i = 0; i < N; i++)
	//	{
	//		for (int j = 0; j < N; j++)
	//			printf("A[%d][%d] = %f  ", i, j, Aarray[k][IDX2C(i, j, N)]);
	//		printf("\n");
	//	}
	//}


	//cudaMalloc((void **)&dev_Aarray, 1 * sizeof(float*)); // �������� ������ �� ������ �� ���������� �� float
	//float **h_tempA = (float **)malloc(1 * sizeof(float*));
	//for (int i = 0; i<1; i++) {
	//	cudaMalloc((void**)&h_tempA[i], N * N * sizeof(float));
	//}
	//for (int i = 0; i<1; i++) {
	//	cudaMemcpy(h_tempA[i], Aarray[i], N * sizeof(float), cudaMemcpyHostToDevice);
	//}
	//cudaMemcpy(dev_Aarray, h_tempA, 1 * sizeof(float*), cudaMemcpyHostToDevice);
	/*printf("A matrix:\n");
	show << <1, 1 >> >(dev_Aarray, N*N, 1);

	printf("Generated matrix Aarray:\n");
	for (int i = 0; i < N; i++)
	{
	for (int j = 0; j < N; j++)
	printf("A[%d][%d] = %f  ", i, j, A[IDX2C(i, j, N)]);
	printf("\n");
	}*/

#pragma endregion
	
	for (int j = 0; j < N; j++)
		printf("B[%d] = %f  ", j, b[j]);
	// �������� ������ �� GPU ���������������� �������
	// ��� ������ ����������
	cudaStat =cudaMalloc((void**)&dev_b, N * sizeof(*x));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		//return EXIT_FAILURE;
	}
	cudaStat =cudaMalloc((void**)&dev_A, N * N * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		//return EXIT_FAILURE;
	}
	//cudaMalloc((void**)&dev_info_array, batch_size* sizeof(int));
	// �������������� �������� cuBLAS
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		//return EXIT_FAILURE;
	}
	// �������� ������ � ������� �� CPU � GPU
	stat =  cublasSetVector(N, sizeof(*b), b, 1, dev_b, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS set vector failed\n");
		//return EXIT_FAILURE;
	}
#pragma region HomeFirst
	//cublasSetVector(batch_size, sizeof(*info_array), info_array, 1, dev_info_array, 1);
	/*cublasSetMatrix(N, N, sizeof(*A), A, N, dev_A, N);

	cublasSgetrfBatched(handle,
	1,
	dev_Aarray,
	6,
	NULL,
	dev_info_array,
	1);*/
#pragma endregion

	stat = cublasSetMatrix(N, N, sizeof(*A), A, N, dev_A, N);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS set matrix failed\n");
		//return EXIT_FAILURE;
	}
	// ������ ������ ���������� �������
	stat = cublasStrsv(handle, CUBLAS_FILL_MODE_LOWER,
		CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, dev_A, N,
		dev_b, 1); 
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS solve failed\n");
		//return EXIT_FAILURE;
	}
	// �������� ��������� �� GPU � CPU
	stat = cublasGetVector(N, sizeof(*x), dev_b, 1, x, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS get vector failed\n");
		//return EXIT_FAILURE;
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("%3.0f ", A[IDX2C(i, j, N)]);
		printf(" = %f %4.6f\n", b[i], x[i]);
	}
	// ����������� ������ � GPU
	cudaFree(dev_b);
	cudaFree(dev_A);
	// ���������� ������� cuBLAS
	cublasDestroy(handle);
	// ����������� ������ � CPU
	free(x);
	free(b);
	free(A);
}

void MultiCublas(int N)
{
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	float *dev_A, *dev_B;
	float *A, *B;
	B = (float *)malloc(N * N * sizeof(*B));
	if (!B) {
		printf("host memory allocation failed");
		//return EXIT_FAILURE;
	}
	A = (float *)malloc(N * N * sizeof(*A));
	if (!A) {
		printf("host memory allocation failed");
		//return EXIT_FAILURE;
	}
	// ������������� ������� � ������� ������ �����
	int ind = 11;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[IDX2C(i, j, N)] = (float)ind++;
			B[IDX2C(i, j, N)] = (float)ind++;
		}
	}
	//printf("A gen:\n");
	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < N; j++)
	//		printf("%0.0f ", A[IDX2C(i, j, N)]);
	//	printf("\n");
	//}
	//printf("B gen:\n");
	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < N; j++)
	//		printf("%0.0f ", B[IDX2C(i, j, N)]);
	//	printf("\n");
	//}

	// �������� ������ �� GPU ���������������� �������
	// ��� ������ ����������
	cudaStat = cudaMalloc((void**)&dev_B, N * N*sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		//return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&dev_A, N * N * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf("device memory allocation failed");
		//return EXIT_FAILURE;
	}

	// �������������� �������� cuBLAS
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		//return EXIT_FAILURE;
	}
	// �������� ������ � ������� �� CPU � GPU
	stat = cublasSetMatrix(N, N, sizeof(*B), B, N, dev_B, N);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS set matrix failed\n");
		//return EXIT_FAILURE;
	}

	stat = cublasSetMatrix(N, N, sizeof(*A), A, N, dev_A, N);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS set matrix failed\n");
		//return EXIT_FAILURE;
	}

	// �������������� �������
	cudaEvent_t start, stop;
	float elapsedTime;
	// ������� �������
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// A*B = A
	float alpha = 1;
	float beta = 0;
	// ������ �������
	cudaEventRecord(start, 0);
	
	cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		N, N, N,
		&alpha,
		dev_A, N,
		dev_B, N,
		&beta,
		dev_A, N);
	cudaEventRecord(stop, 0);
	// �������� ���������� ������ ����
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// ����� ����������
	printf("-----------------------------\n");
	printf("CUBLAS; Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS solve failed\n");
		//return EXIT_FAILURE;
	}
	// �������� ��������� �� GPU � CPU
	
	stat = cublasGetMatrix(N, N, sizeof(*A),
		dev_A, N, A, N);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS get vector failed\n");
		//return EXIT_FAILURE;
	}
	//printf("Res:\n");
	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < N; j++)
	//		printf("%0.0f ", A[IDX2C(i, j, N)]);
	//	printf("\n");
	//}
	// ����������� ������ � GPU
	cudaFree(dev_B);
	cudaFree(dev_A);
	// ���������� ������� cuBLAS
	cublasDestroy(handle);
	// ����������� ������ � CPU
	free(B);
	free(A);
	// ����������� �������
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}

void MultiCublasXt(int N)
{
#pragma region Init
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasXtHandle_t handle;
	int devices[1] = { 0 };
	// �������������� �������� CUBLAS-XT
	stat = cublasXtCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT initialization error\n");
		//return EXIT_FAILURE;
	}
	/* �������� ���������� ��� ������� ������� CUBLAS-XT
	����� �������� ����� �������������� �������� */
	stat = cublasXtDeviceSelect(handle, 1, devices);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT device selection	error\n");
		//return EXIT_FAILURE;
	}
	/* ������������� ������ ������ (blockDim x blockDim) ��
	������� ����� ����������� ������� ��� ������������� �����
	������������ */
	stat = cublasXtSetBlockDim(handle, 64);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT set block dimension error\n");
		//return EXIT_FAILURE;
	}
	float *dev_A, *dev_B;
	float *A, *B, *C;
	B = (float *)malloc(N * N * sizeof(*B));
	if (!B) {
		printf("host memory allocation failed");
		//return EXIT_FAILURE;
	}
	A = (float *)malloc(N * N * sizeof(*A));
	if (!A) {
		printf("host memory allocation failed");
		//return EXIT_FAILURE;
	}
	C = (float *)malloc(N * N * sizeof(*C));
	if (!C) {
		printf("host memory allocation failed");
		//return EXIT_FAILURE;
	}
	// ������������� ������� � ������� ������ �����
	int ind = 11;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[IDX2C(i, j, N)] = (float)ind++;
			B[IDX2C(i, j, N)] = (float)ind++;
		}
	}
	/*printf("A gen:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("%0.0f ", A[IDX2C(i, j, N)]);
		printf("\n");
	}
	printf("B gen:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("%0.0f ", B[IDX2C(i, j, N)]);
		printf("\n");
	}*/
#pragma endregion	
	// �������������� �������
	cudaEvent_t start, stop;
	float elapsedTime;
	// ������� �������
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// A*B = A
	float alpha = 1;
	float beta = 0;
	// ������ �������
	cudaEventRecord(start, 0);

	stat = cublasXtSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		N, N, N,
		&alpha,
		A, N,
		B, N,
		&beta,
		C, N);
	cudaEventRecord(stop, 0);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		//return EXIT_FAILURE;
	}
	// �������� ���������� ������ ����
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// ����� ����������
	printf("-----------------------------\n");
	printf("CUBLASXT; Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);
	printf("-----------------------------\n");
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS solve failed\n");
		//return EXIT_FAILURE;
	}
	//printf("Res:\n");
	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < N; j++)
	//		printf("%0.0f ", C[IDX2C(i, j, N)]);
	//	printf("\n");
	//}
	cublasXtDestroy(handle);
	free(B);
	free(A);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}


__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) 
{
	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;
	float tmpSum = 0;
	
	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		for (int i = 0; i < N; i++) {
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}

		C[ROW * N + COL] = tmpSum;
	}
}

void MultiSimple(int N)
{
	cudaEvent_t start, stop;
	float elapsedTime;
	// ������� �������
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int threads_num = 10;
	int blocks_num = N / threads_num;
	dim3 threadsPerBlock = dim3(threads_num, threads_num);
	dim3 blocksPerGrid = dim3(N/blocks_num, N/blocks_num);

	float *h_A = (float *)malloc(N*N * sizeof(float));
	float *h_B = (float *)malloc(N*N * sizeof(float));
	float *h_C = (float *)malloc(N*N * sizeof(float));
	int ind = 11;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			h_A[IDX2C(i, j, N)] = (float)ind++;
			h_B[IDX2C(i, j, N)] = (float)ind++;
			h_C[IDX2C(i, j, N)] = 0;
		}
	}
	/*printf("A matrix:\n");
	for (int i = 0; i != N; i++)
	{
		for (int j = 0; j != N; j++)
		{
			printf("%.1f ", h_A[i*N + j]);
			printf("%.1f ", h_B[i*N + j]);
		}
		printf("\n");
	}*/

	float *d_A = NULL;
	cudaMalloc((void **)&d_A, N*N * sizeof(float));
	float * d_B = NULL;
	cudaMalloc((void **)&d_B, N*N * sizeof(float));
	float * d_C = NULL;
	cudaMalloc((void **)&d_C, N*N * sizeof(float));

	cudaMemcpy(d_A, h_A, N*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N*N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, N*N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(h_C, d_C, N*N * sizeof(float), cudaMemcpyDeviceToHost);
	/*for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%f ", h_C[IDX2C(i, j, N)]);
		}
		printf("\n");
	}*/
	printf("Simple; Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);
	printf("-----------------------------\n");
	free(h_B);
	free(h_A);
	free(h_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void ThirdHome()
{
	for (int i = 10; i <= 100; i += 10)
	{
		printf("N = %d\n", i);
		MultiCublas(i);
		MultiCublasXt(i);
		MultiSimple(i);
	}
}

#define N (2)
void FirstLab()
{
#pragma region Init CuBlasXt
	cublasStatus_t status;
	cublasXtHandle_t handle;
	int devices[1] = { 0 };
	// �������������� �������� CUBLAS-XT
	status = cublasXtCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT initialization error\n");
		//return EXIT_FAILURE;
	}
	/* �������� ���������� ��� ������� ������� CUBLAS-XT
	����� �������� ����� �������������� �������� */
	status = cublasXtDeviceSelect(handle, 1, devices);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT device selection	error\n");
		//return EXIT_FAILURE;
	}
	/* ������������� ������ ������ (blockDim x blockDim) ��
	������� ����� ����������� ������� ��� ������������� �����
	������������ */
	status = cublasXtSetBlockDim(handle, 64);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT set block dimension error\n");
		//return EXIT_FAILURE;
	}
#pragma endregion


	float *h_A;
	float *h_B;
	float *h_I;
	float *h_ABSum;
	float *h_C;
	// �������
	float alpha = 1.0f;
	float beta = 1.0f;
	// ������ �������
	int n2 = N * N;
	
	// �������� ������ ��� ������ � ��������� ������
	h_A = (float *)malloc(n2 * sizeof(h_A[0]));
	h_B = (float *)malloc(n2 * sizeof(h_B[0]));
	h_ABSum = (float *)malloc(n2 * sizeof(h_ABSum[0]));
	h_I = (float *)malloc(n2 * sizeof(h_I[0]));
	h_C = (float *)malloc(n2 * sizeof(h_C[0]));
	// ��������� ������� ��������� �������
	srand((unsigned int)time(NULL));
	for (int i = 0; i < n2; i++)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
		h_ABSum[i] = h_B[i];
	}
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			h_I[i*N+j] = i == j ? 1 : 0;

#pragma region Print
	printf("initialized A:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("A[%d][%d] = %f  ", i, j, h_A[i*N + j]);
		printf("\n");
	}
	printf("initialized B:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("B[%d][%d] = %f  ", i, j, h_B[i*N + j]);
		printf("\n");
	}
	printf("initialized I:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("I[%d][%d] = %f  ", i, j, h_I[i*N + j]);
		printf("\n");
	}
#pragma endregion


	// ��������� �������� ������������ ������
	printf("1*A*I + 1*B :\n");
	status = cublasXtSgemm(handle, CUBLAS_OP_N,
		CUBLAS_OP_N, N, N, N, &alpha, h_A, N, h_I, N, &beta,
		h_ABSum, N);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! kernel execution error.\n");
		//return EXIT_FAILURE;
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("ABSum[%d][%d] = %f  ", i, j, h_ABSum[i*N + j]);
		printf("\n");
	}
	beta = 0.0f;
	// ��������� �������� ������������ ������
	printf("1*A*B + 0*C :\n");
	status = cublasXtSgemm(handle, CUBLAS_OP_N,
		CUBLAS_OP_N, N, N, N, &alpha, h_A, N, h_B, N, &beta,
		h_C, N);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! kernel execution error.\n");
		//return EXIT_FAILURE;
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("C[%d][%d] = %f  ", i, j, h_C[i*N + j]);
		printf("\n");
	}
	// ��������� �������� ������������ ������
	printf("1*(A+B) * (A*B) + 0*C :\n");
	status = cublasXtSgemm(handle, CUBLAS_OP_N,
		CUBLAS_OP_N, N, N, N, &alpha, h_ABSum, N, h_C, N, &beta,
		h_C, N);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! kernel execution error.\n");
		//return EXIT_FAILURE;
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("C[%d][%d] = %f  ", i, j, h_C[i*N + j]);
		printf("\n");
	}
	
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_ABSum);
}


void SecondHome()
{
#pragma region Init CuBlasXt
	cublasStatus_t status;
	cublasXtHandle_t handle;
	int devices[1] = { 0 };
	// �������������� �������� CUBLAS-XT
	status = cublasXtCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT initialization error\n");
		//return EXIT_FAILURE;
	}
	/* �������� ���������� ��� ������� ������� CUBLAS-XT
	����� �������� ����� �������������� �������� */
	status = cublasXtDeviceSelect(handle, 1, devices);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT device selection	error\n");
		//return EXIT_FAILURE;
	}
	/* ������������� ������ ������ (blockDim x blockDim) ��
	������� ����� ����������� ������� ��� ������������� �����
	������������ */
	status = cublasXtSetBlockDim(handle, 64);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLASXT set block dimension error\n");
		//return EXIT_FAILURE;
	}
#pragma endregion


	float *h_A;
	float *h_B;
	float *h_I;
	float *h_ACSum;
	float *h_C;
	// �������
	float alpha = 1.0f;
	float beta = 1.0f;
	// ������ �������
	int n2 = N * N;

	// �������� ������ ��� ������ � ��������� ������
	h_A = (float *)malloc(n2 * sizeof(h_A[0]));
	h_B = (float *)malloc(n2 * sizeof(h_B[0]));
	h_ACSum = (float *)malloc(n2 * sizeof(h_ACSum[0]));
	h_I = (float *)malloc(n2 * sizeof(h_I[0]));
	h_C = (float *)malloc(n2 * sizeof(h_C[0]));
	// ��������� ������� ��������� �������
	srand((unsigned int)time(NULL));

	h_A[0] = 1; h_A[1] = 1; h_A[2] = 0; h_A[3] = 1;
	h_C[0] = 1; h_C[1] = -1; h_C[2] = 0; h_C[3] = 1;
	h_B[0] = 12; h_B[1] = 10; h_B[2] = 6; h_B[3] = 8;

	for (int i = 0; i < n2; i++)
	{
		h_ACSum[i] = h_C[i];
	}
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			h_I[i*N + j] = i == j ? 1 : 0;

#pragma region Print
	printf("initialized A:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("A[%d][%d] = %f  ", i, j, h_A[i*N + j]);
		printf("\n");
	}
	printf("initialized B:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("B[%d][%d] = %f  ", i, j, h_B[i*N + j]);
		printf("\n");
	}
	printf("initialized C:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("C[%d][%d] = %f  ", i, j, h_C[i*N + j]);
		printf("\n");
	}
	printf("initialized I:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("I[%d][%d] = %f  ", i, j, h_I[i*N + j]);
		printf("\n");
	}
#pragma endregion


	// ��������� �������� ������������ ������
	printf("1*A*I + 1*C :\n");
	status = cublasXtSgemm(handle, CUBLAS_OP_N,
		CUBLAS_OP_N, N, N, N, &alpha, h_A, N, h_I, N, &beta,
		h_ACSum, N);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! kernel execution error.\n");
		//return EXIT_FAILURE;
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("ACSum[%d][%d] = %f  ", i, j, h_ACSum[i*N + j]);
		printf("\n");
	}
	beta = 0.0f;
	// ��������� �������� ������������ ������
	printf("1*(A*C)*C + 0*C :\n");
	status = cublasXtSgemm(handle, CUBLAS_OP_N,
		CUBLAS_OP_N, N, N, N, &alpha, h_ACSum, N, h_C, N, &beta,
		h_C, N);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! kernel execution error.\n");
		//return EXIT_FAILURE;
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("C[%d][%d] = %f  ", i, j, h_C[i*N + j]);
		printf("\n");
	}
	// ��������� ������ ������� �
	printf("1*(A+B) * (A*B) + 0*C :\n");
	cublasXtStrsm(handle,
		CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
		N, N,
		&alpha,
		h_C, N,
		h_B, N);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			printf("X[%d][%d] = %f  ", i, j, h_B[i*N + j]);
		printf("\n");
	}

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! kernel execution error.\n");
		//return EXIT_FAILURE;
	}

	free(h_A);
	free(h_B);
	free(h_C);
	free(h_ACSum);
}


int main()
{	
	//FirstLab();
	//SecondHome();
	
	//CublasExample();
	ThirdHome();
	getchar();
	return EXIT_SUCCESS;
}