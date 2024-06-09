
#pragma once

extern "C"
{
	void cudaInit();
	void cudaReset();
	void cudaSynchronize();
	void allocateDevice(void** d_ptr, int size);
	void freeDevice(void* d_ptr);
	void copyDeviceToHost(const void* device, void* host, int size);
	void copyHostToDevice(void* device, const void* host, int size);
	void copyDeviceToDevice(void* src, void* dst, int size);
	void copyParametersToDevice(const void* host, int size);
	void registerGLBufferObject(int vbo, cudaGraphicsResource** res);
	void unregisterGLBufferObject(cudaGraphicsResource* res);
	void* mapGLBufferObject(cudaGraphicsResource** res);
	void unmapGLBufferObject(cudaGraphicsResource* res);

	void curandStateSetup(curandState* d_curandState, uint64_t seed, int numBoids);
	void updateBoids(float2* newPos, float2* newVel, int numBoids);
	void calculateIndexes(int* cellInd, int* boidInd, float2* currentPos, int numBoids);
	void sortBoids(int* cellInd, int* boidInd, int numBoids);
	void findCellStartAndEnd(int* cellStart, int* cellEnd, int* cellInd, int numBoids, int numCells);
	void steerBoids(float2* newVel, float2* currentPos, float2* oldVel, int* boidInd, int* cellStart, int* cellEnd, int numBoids, int numCells, curandState* state);
	void naiveSteerBoids(float2* newVel, float2* currentPos, float2* oldVel, int numBoids, curandState* state);
}