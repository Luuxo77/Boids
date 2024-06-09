
#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <math.h>

inline __host__ __device__ float2 operator+(float2 a, float b)
{
	return make_float2(a.x + b, a.y + b);
}

inline __host__ __device__ float2 operator+(float a, float2 b)
{
	return make_float2(a + b.x, a + b.y);
}

inline __host__ __device__ void operator+=(float2& a, float b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ float2 operator-(float2 a, float b)
{
	return make_float2(a.x - b, a.y - b);
}

inline __host__ __device__ void operator-=(float2& a, float b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float2 operator*(float a, float2 b)
{
	return make_float2(a * b.x, a * b.y);
}

inline __host__ __device__ void operator*=(float2& a, float b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ float2 operator/(float2 a, float b)
{
	return make_float2(a.x / b, a.y / b);
}

inline __host__ __device__ void operator/=(float2& a, float b)
{
	a.x /= b;
	a.y /= b;
}

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(float2& a, float2 b)
{
	a.x += b.x;
	a.y += b.y;
}

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(float2& a, float2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}

inline __host__ __device__ float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float length(float2 a)
{
	return sqrtf(dot(a, a));
}

inline __host__ __device__ float2 make_float2(int a, int b)
{
	return make_float2((float)a, (float)b);
}

inline __host__ __device__ void clamp(float2& value, float min, float max)
{
	float len = length(value);
	if (len > max)
		value = (value / len) * max;
	else if (len < min)
		value = (value / len) * min;
}

inline __host__ __device__ bool operator==(float2 a, float2 b)
{
	return a.x == b.x && a.y == b.y;
}