
#pragma once

#include <vector_types.h>

struct Parameters
{
	int numBoids;
	int numCells;
	int2 boxSize;
	int2 gridSize;
	float2 cellSize;
	int algorithm;
	bool mouseDown;
	float2 mousePosition;
	float cohesionFactor;
	float allignmentFactor;
	float separationFactor;
	float escapeFactor;
	float margin;
	float marginFactor;
	float viewRange;
	float minDist;
	float minSpeed;
	float maxSpeed;
};