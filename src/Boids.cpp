
#include "Parameters.cuh"
#include "Boids.h"
#include "kernels.cuh"

Boids::Boids(Parameters _params) : params(_params)
{
	copyParamsToDevice();
	srand((unsigned int)time(0));

	h_pos = new float2[params.numBoids];
	h_vel = new float2[params.numBoids];
	h_oldVel = new float2[params.numBoids];
	resetData();

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &posVBO);
	glGenBuffers(1, &velVBO);
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * params.numBoids, h_pos, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);


	glBindBuffer(GL_ARRAY_BUFFER, velVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * params.numBoids, h_vel, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glBindVertexArray(0);

	cudaInit();
	allocateDevice((void**)&d_oldVel, sizeof(float2) * params.numBoids);
	allocateDevice((void**)&d_cellInd, sizeof(int) * params.numBoids);
	allocateDevice((void**)&d_particleInd, sizeof(int) * params.numBoids);
	allocateDevice((void**)&d_cellStart, sizeof(int) * params.numCells);
	allocateDevice((void**)&d_cellEnd, sizeof(int) * params.numCells);
	allocateDevice((void**)&d_curandState, sizeof(curandState) * params.numBoids);

	curandStateSetup(d_curandState, time(0), params.numBoids);

	registerGLBufferObject(posVBO, &cuda_posVBO);
	registerGLBufferObject(velVBO, &cuda_velVBO);
	cudaSynchronize();
}

void Boids::compute()
{
	switch (params.algorithm)
	{
	case 0:
	{
		computeGPU_Grid();
		break;
	}
	case 1:
	{
		computeGPU_Naive();
		break;
	}
	case 2:
	{
		computeCPU();
		break;
	}
	}
}

void Boids::computeGPU_Grid()
{
	float2* d_pos = (float2*)mapGLBufferObject(&cuda_posVBO);
	float2* d_vel = (float2*)mapGLBufferObject(&cuda_velVBO);
	calculateIndexes(d_cellInd, d_particleInd, d_pos, params.numBoids);
	sortBoids(d_cellInd, d_particleInd, params.numBoids);
	findCellStartAndEnd(d_cellStart, d_cellEnd, d_cellInd, params.numBoids, params.numCells);
	copyDeviceToDevice(d_vel, d_oldVel, params.numBoids * sizeof(float2));
	steerBoids(d_vel, d_pos, d_oldVel, d_particleInd, d_cellStart, d_cellEnd, params.numBoids, params.numCells, d_curandState);
	updateBoids(d_pos, d_vel, params.numBoids);
	unmapGLBufferObject(cuda_velVBO);
	unmapGLBufferObject(cuda_posVBO);
	cudaSynchronize();
}

void Boids::computeGPU_Naive()
{
	float2* d_pos = (float2*)mapGLBufferObject(&cuda_posVBO);
	float2* d_vel = (float2*)mapGLBufferObject(&cuda_velVBO);
	copyDeviceToDevice(d_vel, d_oldVel, params.numBoids * sizeof(float2));
	naiveSteerBoids(d_vel, d_pos, d_oldVel, params.numBoids, d_curandState);
	updateBoids(d_pos, d_vel, params.numBoids);
	unmapGLBufferObject(cuda_velVBO);
	unmapGLBufferObject(cuda_posVBO);
	cudaSynchronize();
}

void Boids::computeCPU()
{
	memcpy(h_oldVel, h_vel, params.numBoids * sizeof(float2));
	steerBoidsCPU();
	updateBoidsCPU();
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float2) * params.numBoids, h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, velVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float2) * params.numBoids, h_vel);
}

void Boids::render()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, params.numBoids);
	glBindVertexArray(0);
}

void Boids::reset()
{
	copyParamsToDevice();
	resetData();
	memcpy(h_oldVel, h_vel, sizeof(float2) * params.numBoids);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float2) * params.numBoids, h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, velVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float2) * params.numBoids, h_vel);
}

void Boids::resetData()
{
	for (int i = 0; i < params.numBoids; i++)
	{
		float x = ((float)rand() / (float)RAND_MAX) * (float)params.boxSize.x;
		float y = ((float)rand() / (float)RAND_MAX) * (float)params.boxSize.y;
		h_pos[i] = make_float2(x, y);
		float vx = ((float)rand() / (float)RAND_MAX) * params.maxSpeed * 2 - params.maxSpeed;
		float vy = ((float)rand() / (float)RAND_MAX) * params.maxSpeed * 2 - params.maxSpeed;
		h_vel[i] = make_float2(vx, vy);
	}
}

void Boids::resetParams()
{
	params.cohesionFactor = 0.0005f;
	params.allignmentFactor = 0.05f;
	params.separationFactor = 0.05f;
}

void Boids::changeBoxSize(int2 boxSize)
{
	params.boxSize = boxSize;
	params.cellSize = make_float2
	(
		(float)params.boxSize.x / params.gridSize.x,
		(float)params.boxSize.y / params.gridSize.y
	);
	copyParamsToDevice();
}

void Boids::copyParamsToDevice()
{
	copyParametersToDevice(&params, sizeof(Parameters));
}


void Boids::updateBoidsCPU()
{
	for (int i = 0; i < params.numBoids; i++)
	{
		float2 pos = h_pos[i];
		float2 vel = h_vel[i];

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

		h_pos[i] = pos;
		h_vel[i] = vel;
	}
}

void Boids::steerBoidsCPU()
{
	for (int i = 0; i < params.numBoids; i++)
	{
		float2 pos = h_pos[i];
		float2 vel = h_oldVel[i];

		float2 centerPos = make_float2(0, 0);
		float2 averageVel = make_float2(0, 0);
		float2 separationDir = make_float2(0, 0);
		float2 additionalVel = make_float2(0, 0);
		int neighbours = 0;

		for (int j = 0; j < params.numBoids; j++)
			if (j != i)
			{
				float2 otherPos = h_pos[j];
				float2 toOther = otherPos - pos;
				float dist = length(toOther);
				float2 otherVel = h_oldVel[j];
				if (dist == 0.0f && vel == otherVel && params.separationFactor != 0.0f)
				{
					separationDir.x += ((float)rand() / (float)RAND_MAX) - 0.5f;
					separationDir.y += ((float)rand() / (float)RAND_MAX) - 0.5f;
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
			centerPos /= (float)neighbours;
			additionalVel += (centerPos - pos) * params.cohesionFactor;
			averageVel /= (float)neighbours;
			additionalVel += (averageVel - vel) * params.allignmentFactor;
		}
		additionalVel += separationDir * params.separationFactor;

		float2 newV = vel + additionalVel;
		clamp(newV, params.minSpeed, params.maxSpeed);
		h_vel[i] = newV;
	}
}

Boids::~Boids()
{
	delete[] h_pos;
	delete[] h_vel;
	delete[] h_oldVel;
	freeDevice(d_oldVel);
	freeDevice(d_cellInd);
	freeDevice(d_particleInd);
	freeDevice(d_cellStart);
	freeDevice(d_cellEnd);
	freeDevice(d_curandState);
	unregisterGLBufferObject(cuda_posVBO);
	unregisterGLBufferObject(cuda_velVBO);
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &posVBO);
	glDeleteBuffers(1, &velVBO);
	cudaReset();
}