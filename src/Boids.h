
#pragma once

#include "Parameters.cuh"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <shader.h>
#include "helper.h"
#include "cuda_gl_interop.h"
#include <curand_kernel.h>

class Boids
{
public:
	float2* h_pos = 0;
	float2* h_vel = 0;
	float2* h_oldVel = 0;

	float2* d_oldVel = 0;
	int* d_cellStart = 0;
	int* d_cellEnd = 0;
	int* d_particleInd = 0;
	int* d_cellInd = 0;
	curandState* d_curandState = 0;

	unsigned int posVBO;
	unsigned int velVBO;
	unsigned int VAO;

	cudaGraphicsResource* cuda_posVBO = 0;
	cudaGraphicsResource* cuda_velVBO = 0;

	Parameters params;

	Boids(Parameters par);
	~Boids();
	void render();
	void compute();
	void computeGPU_Grid();
	void computeGPU_Naive();
	void computeCPU();
	void updateBoidsCPU();
	void steerBoidsCPU();
	void reset();
	void resetData();
	void resetParams();
	void changeBoxSize(int2 boxSize);
	void copyParamsToDevice();
};