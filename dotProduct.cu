#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <time.h>
#include <sstream>
#include <iostream>

#ifndef __CUDACC_RTC__
#define __CUDACC_RTC__
#endif

#include <device_functions.h>

using namespace std;

#define imin(a, b) (a<b? a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {

	__shared__ float cache[threadsPerBlock];
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0.0;

	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = temp;

	__syncthreads();

	int i = blockDim.x / 2;

	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i];
			}
		__syncthreads();
		i = i / 2;
	}

	if (cacheIndex == 0) {
		c[blockIdx.x] = cache[0];
	}
}

int main(void) {
	float *a, *b, *c, result, cpu_result;
	float *dev_a, *dev_b, *dev_c;
	

	a = new float[N];
	b = new float[N];
	c = new float[N];
	result = 0;
	cpu_result = 0;
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
		c[i] = 0;
	}
	cout << "start" << endl;
	
	for (int i = 0; i < N; i++) {
		cpu_result += a[i] * b[i];
	}

	
	cout << cpu_result << endl;

	cudaMalloc((void**)&dev_a, sizeof(float)*N);
	cudaMalloc((void**)&dev_b, sizeof(float)*N);
	cudaMalloc((void**)&dev_c, sizeof(float)*blocksPerGrid);

	cudaMemcpy(dev_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemset(dev_c, 0, sizeof(float)*blocksPerGrid);


	dot <<< blocksPerGrid, threadsPerBlock >>> (dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, sizeof(float)*blocksPerGrid, cudaMemcpyDeviceToHost);

	for (int j = 0; j < blocksPerGrid; j++)
	{
		result += c[j];
	}


	std::cout << result << std::endl;

	system("pause");

	cudaDeviceReset();

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}