
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "kernels.cuh"
#include "Parameters.cuh"
#include "helper.h"

#define NUMBER_OF_THREADS 512

__constant__ Parameters params;

void checkErrors(cudaError_t err, const char* name)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s failed!", name);
		exit(-1);
	}
}

void cudaInit()
{
	checkErrors(cudaSetDevice(0), "cudaSetDevice");
}

void cudaReset()
{
	checkErrors(cudaDeviceReset(), "cudaDeviceReset");
}

void cudaSynchronize()
{
	checkErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void allocateDevice(void** d_ptr, int size)
{
	checkErrors(cudaMalloc(d_ptr, size), "cudaMalloc");
}

void freeDevice(void* d_ptr)
{
	checkErrors(cudaFree(d_ptr), "cudaFree");
}

void copyHostToDevice(void* d_ptr, const void* h_ptr, int size)
{
	checkErrors(cudaMemcpy((void*)d_ptr, h_ptr, size, cudaMemcpyHostToDevice), "cudaMemcpy");
}

void copyDeviceToHost(const void* d_ptr, void* h_ptr, int size)
{
	checkErrors(cudaMemcpy((void*)h_ptr, d_ptr, size, cudaMemcpyDeviceToHost), "cudaMemcpy");
}

void copyDeviceToDevice(void* d_src, void* d_dst, int size)
{
	checkErrors(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

void copyParametersToDevice(const void* h_ptr, int size)
{
	checkErrors(cudaMemcpyToSymbol(params, h_ptr, size, 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
}

void registerGLBufferObject(int vbo, cudaGraphicsResource** res)
{
	checkErrors(cudaGraphicsGLRegisterBuffer(res, vbo, cudaGraphicsMapFlagsNone), "cudaGraphicsGLRegisterBuffer");
}

void unregisterGLBufferObject(cudaGraphicsResource* res)
{
	checkErrors(cudaGraphicsUnregisterResource(res), "cudaGraphicsUnregisterResource");
}

void* mapGLBufferObject(cudaGraphicsResource** res)
{
	void* ptr = 0;
	checkErrors(cudaGraphicsMapResources(1, res, 0), "cudaGraphicsMapResources");
	checkErrors(cudaGraphicsResourceGetMappedPointer((void**)&ptr, 0, *res), "cudaGraphicsResourceGetMappedPointer");
	return ptr;
}

void unmapGLBufferObject(cudaGraphicsResource* res)
{
	checkErrors(cudaGraphicsUnmapResources(1, &res, 0), "cudaGraphicsUnmapResources");
}

int calculateBlocks(int numBoids, int threadsInBlock)
{
	return (numBoids % threadsInBlock != 0) ? (numBoids / threadsInBlock + 1) : (numBoids / threadsInBlock);
}

inline __device__ int calculateGridIndex(float2 pos)
{
	int2 cellIndex{};
	cellIndex.x = floorf(pos.x / params.cellSize.x);
	cellIndex.y = floorf(pos.y / params.cellSize.y);
	return cellIndex.x + params.gridSize.x * cellIndex.y;
}

__global__ void curandStateSetupKernel(curandState* d_curandState, uint64_t seed)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= params.numBoids)
		return;
	curand_init(seed, index, 0, &d_curandState[index]);
}

void curandStateSetup(curandState* d_curandState, uint64_t seed, int numBoids)
{
	int threadsInBlock = std::min(numBoids, NUMBER_OF_THREADS);
	int numBlocks = calculateBlocks(numBoids, threadsInBlock);
	curandStateSetupKernel << <numBlocks, threadsInBlock >> > (d_curandState, seed);
	checkErrors(cudaGetLastError(), "curandStateSetup launch");
}

__global__ void updateBoidsKernel(float2* newPos, float2* newVel)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBoids)
		return;

	float2 pos = newPos[index];
	float2 vel = newVel[index];

	pos += vel;
	float2 size = make_float2(params.boxSize.x, params.boxSize.y);

	if (pos.x > size.x - params.margin)
	{
		vel.x -= params.marginFactor;
		if (pos.x > size.x)
			pos.x = 2 * size.x - pos.x;
	}

	else if (pos.x < params.margin)
	{
		vel.x += params.marginFactor;
		if (pos.x < 0)
			pos.x = -pos.x;
	}

	if (pos.y > size.y - params.margin)
	{
		vel.y -= params.marginFactor;
		if (pos.y > size.y)
			pos.y = 2 * size.y - pos.y;
	}

	else if (pos.y < params.margin)
	{
		vel.y += params.marginFactor;
		if (pos.y < 0)
			pos.y = -pos.y;
	}

	newPos[index] = pos;
	newVel[index] = vel;
}

void updateBoids(float2* newPos, float2* newVel, int numBoids)
{
	int threadsInBlock = std::min(numBoids, NUMBER_OF_THREADS);
	int numBlocks = calculateBlocks(numBoids, threadsInBlock);
	updateBoidsKernel << <numBlocks, threadsInBlock >> > (newPos, newVel);
	checkErrors(cudaGetLastError(), "updateBoidsWallKernel launch");
}

__global__ void calculateIndexesKernel(int* cellInd, int* boidInd, float2* currPos)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBoids)
		return;

	float2 pos = currPos[index];
	int gridIndex = calculateGridIndex(pos);
	cellInd[index] = gridIndex;
	boidInd[index] = index;
}

void calculateIndexes(int* cellInd, int* boidInd, float2* currPos, int numBoids)
{
	int threadsInBlock = std::min(numBoids, NUMBER_OF_THREADS);
	int numBlocks = calculateBlocks(numBoids, threadsInBlock);
	calculateIndexesKernel << <numBlocks, threadsInBlock >> > (cellInd, boidInd, currPos);
	checkErrors(cudaGetLastError(), "calculateIndexesKernel launch");
}

void sortBoids(int* cellInd, int* boidInd, int numBoids)
{
	thrust::sort_by_key
	(
		thrust::device_ptr<int>(cellInd),
		thrust::device_ptr<int>(cellInd + numBoids),
		thrust::device_ptr<int>(boidInd)
	);
}

__global__ void findCellStartAndEndKernel(int* cellStart, int* cellEnd, int* cellInd)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBoids)
		return;

	int cellIndex = cellInd[index];
	int prevCellIndex = index > 0 ? cellInd[index - 1] : 0;

	if (index == 0 || cellIndex != prevCellIndex)
	{
		cellStart[cellIndex] = index;
		if (index > 0)
			cellEnd[prevCellIndex] = index;
	}

	if (index == params.numBoids - 1)
		cellEnd[cellIndex] = index + 1;
}

void findCellStartAndEnd(int* cellStart, int* cellEnd, int* cellInd, int numBoids, int numCells)
{
	int threadsInBlock = std::min(numBoids, NUMBER_OF_THREADS);
	int numBlocks = calculateBlocks(numBoids, threadsInBlock);
	checkErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(int)), "cudaMemset");
	findCellStartAndEndKernel << <numBlocks, threadsInBlock >> > (cellStart, cellEnd, cellInd);
	checkErrors(cudaGetLastError(), "findCellStartAndEnd launch");
}

__global__ void steerBoidsKernel(float2* newVel, float2* currentPos, float2* oldVel,
	int* boidInd, int* cellStart, int* cellEnd, curandState* state)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBoids)
		return;

	int boidIndex = boidInd[index];

	float2 pos = currentPos[boidIndex];
	float2 vel = oldVel[boidIndex];

	int gridIndex = calculateGridIndex(pos);
	int cellsInRow = params.gridSize.x;

	float2 centerPos = make_float2(0, 0);
	float2 averageVel = make_float2(0, 0);
	float2 separationDir = make_float2(0, 0);
	float2 additionalVel = make_float2(0, 0);
	int neighbours = 0;

	for (int y = -1; y <= 1; y++)
		for (int x = -1; x <= 1; x++)
		{
			int nextGridIndex = gridIndex + x + y * cellsInRow;
			if (nextGridIndex >= 0 && nextGridIndex < params.numCells)
			{
				int startIndex = cellStart[nextGridIndex];
				if (startIndex != 0xffffffff)
				{
					int endIndex = cellEnd[nextGridIndex];
					for (int i = startIndex; i < endIndex; i++)
					{
						int otherBoidIndex = boidInd[i];
						if (otherBoidIndex != boidIndex)
						{
							float2 otherPos = currentPos[otherBoidIndex];
							float2 toOther = otherPos - pos;
							float dist = length(toOther);
							float2 otherVel = oldVel[otherBoidIndex];
							if (dist == 0.0f && vel == otherVel && params.separationFactor != 0.0f)
							{
								curandState localState = state[index];
								separationDir.x += curand_uniform(&localState) - 0.5f;
								separationDir.y += curand_uniform(&localState) - 0.5f;
							}
							if (dist < params.viewRange)
							{
								if (dist < params.minDist)
									separationDir -= toOther;
								centerPos += otherPos;
								averageVel += otherVel;
								neighbours++;
							}
						}
					}
				}
			}
		}

	if (params.mouseDown)
	{
		float2 toMouse = params.mousePosition - pos;
		float dist = length(toMouse);
		if (dist < params.viewRange)
			additionalVel += (pos - params.mousePosition) * params.escapeFactor;
	}

	if (neighbours)
	{
		centerPos /= neighbours;
		additionalVel += (centerPos - pos) * params.cohesionFactor;
		averageVel /= neighbours;
		additionalVel += (averageVel - vel) * params.allignmentFactor;
	}
	additionalVel += separationDir * params.separationFactor;

	float2 newV = vel + additionalVel;
	clamp(newV, params.minSpeed, params.maxSpeed);
	newVel[boidIndex] = newV;
}

void steerBoids(float2* newVel, float2* currentPos, float2* oldVel, int* boidInd,
	int* cellStart, int* cellEnd, int numBoids, int numCells, curandState* state)
{
	int threadsInBlock = std::min(numBoids, NUMBER_OF_THREADS);
	int numBlocks = calculateBlocks(numBoids, threadsInBlock);
	steerBoidsKernel << < numBlocks, threadsInBlock >> > (newVel, currentPos, oldVel, boidInd, cellStart, cellEnd, state);
	checkErrors(cudaGetLastError(), "steerBoidsKernel launch");
}

__global__ void naiveSteerBoidsKernel(float2* newVel, float2* currentPos, float2* oldVel, curandState* state)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBoids)
		return;

	float2 pos = currentPos[index];
	float2 vel = oldVel[index];

	float2 centerPos = make_float2(0, 0);
	float2 averageVel = make_float2(0, 0);
	float2 separationDir = make_float2(0, 0);
	float2 additionalVel = make_float2(0, 0);
	int neighbours = 0;

	for (int otherIndex = 0; otherIndex < params.numBoids; otherIndex++)
		if (otherIndex != index)
		{
			float2 otherPos = currentPos[otherIndex];
			float2 toOther = otherPos - pos;
			float dist = length(toOther);
			float2 otherVel = oldVel[otherIndex];
			if (dist == 0.0f && vel == otherVel && params.separationFactor != 0.0f)
			{
				curandState localState = state[index];
				separationDir.x += curand_uniform(&localState) - 0.5f;
				separationDir.y += curand_uniform(&localState) - 0.5f;
			}
			if (dist < params.viewRange)
			{
				if (dist < params.minDist)
					separationDir -= toOther;
				centerPos += otherPos;
				averageVel += otherVel;
				neighbours++;
			}
		}

	if (params.mouseDown)
	{
		float2 toMouse = params.mousePosition - pos;
		float dist = length(toMouse);
		if (dist < params.viewRange)
			additionalVel += (pos - params.mousePosition) * params.escapeFactor;
	}

	if (neighbours)
	{
		centerPos /= neighbours;
		additionalVel += (centerPos - pos) * params.cohesionFactor;
		averageVel /= neighbours;
		additionalVel += (averageVel - vel) * params.allignmentFactor;
	}
	additionalVel += separationDir * params.separationFactor;

	float2 newV = vel + additionalVel;
	clamp(newV, params.minSpeed, params.maxSpeed);
	newVel[index] = newV;
}

void naiveSteerBoids(float2* newVel, float2* currentPos, float2* oldVel, int numBoids, curandState* state)
{
	int threadsInBlock = std::min(numBoids, NUMBER_OF_THREADS);
	int numBlocks = calculateBlocks(numBoids, threadsInBlock);
	naiveSteerBoidsKernel << < numBlocks, threadsInBlock >> > (newVel, currentPos, oldVel, state);
	checkErrors(cudaGetLastError(), "naiveSteerBoidsKernel launch");
}