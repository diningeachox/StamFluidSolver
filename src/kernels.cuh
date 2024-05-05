#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <initializer_list>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "density.h"

__global__ void diffuse_kernel(float* d_x, float* d_x0, float diff, float dt, int N);
__global__ void diffuse_kernel_sm(float* d_out, float* d_in, float* d_x0, float diff, float dt, int N);
__global__ void set_bnd_kernel(float* d_out, float* d_in, int b, int N);

void diffuse_parallel(std::vector<float>& x, std::vector<float>& x0, float diff, float dt, int b, int N);