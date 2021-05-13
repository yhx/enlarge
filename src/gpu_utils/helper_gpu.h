/* This header file is writen by qp09
 * usually just for fun
 * Thu February 16 2017
 */
#ifndef HELPER_GPU_H
#define HELPER_GPU_H

#include <assert.h>

#include "../third_party/cuda/helper_cuda.h"

inline void gpuDevice(int device = 0) {
    checkCudaErrors(cudaSetDevice(device));
}

template<typename T>
T* hostMalloc(size_t size = 1)
{
	T * ret;
	checkCudaErrors(cudaMallocHost((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(T)*(size)));
	return ret;
}

template<typename T>
T* gpuMalloc(size_t size = 1)
{ 
	T * ret;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(T)*(size)));
	return ret;
}

template<typename T>
void gpuMemset(T* array, int c, size_t size = 1)
{ 
	assert(array);
	checkCudaErrors(cudaMemset(array, 0, sizeof(T)*(size)));
}

template<typename T>
void hostFree(T* cpu)
{
	assert(cpu);
	checkCudaErrors(cudaFreeHost(cpu));
}

template<typename T>
void gpuFree(T* gpu)
{
	assert(gpu);
	checkCudaErrors(cudaFree(gpu));
}

template<typename T>
void gpuMemcpyPeer(T*data_d, int dst, T*data_s, int src, size_t size = 1)
{
	assert(data_d);
	assert(data_s);
	checkCudaErrors(cudaMemcpyPeer(data_d, dst, data_s, src, sizeof(T)*(size)));
}

template<typename T>
T* copyToGPU(T* cpu, size_t size = 1)
{
	T * ret;
	assert(cpu);
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(T) * size));
	checkCudaErrors(cudaMemcpy(ret, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));

	return ret;
}

template<typename T>
void copyToGPU(T* gpu, T* cpu, size_t size = 1)
{
	assert(cpu);
	assert(gpu);
	checkCudaErrors(cudaMemcpy(gpu, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));
}

template<typename T>
T* copyFromGPU(T* gpu, size_t size = 1)
{
	assert(gpu);
	T * ret = static_cast<T*>(malloc(sizeof(T)*size));
	assert(ret);
	checkCudaErrors(cudaMemcpy(ret, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));

	return ret;
}

template<typename T>
void copyFromGPU(T* cpu, T* gpu, size_t size = 1)
{
	assert(cpu);
	assert(gpu);
	checkCudaErrors(cudaMemcpy(cpu, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));
}

#endif /* HELPER_GPU_H */

