#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <initializer_list>
#include <vector>
#include <stdio.h>
#include "density.h"

#define TPB 64 //threads per block
#define RAD 1 //radius of stencil

__global__ void diffuse_kernel(float* d_x, float* d_x0, float diff, float dt, int N) {
	const float a = dt * diff * N * N;
	//Global index
	const int i = threadIdx.x + blockDim.x * blockIdx.x;
	const int j = threadIdx.y + blockDim.y * blockIdx.y;

	if (i >= N + 2 || j >= N + 2) return;

	//Local index 
	//Each shared memory block is (blockDim.x + 2) * (blockDim.y + 2)
	// Dynamically declared shared memory
	
	extern __shared__ float s_in[];	

	//const int s_idx = threadIdx.x + RAD; 

	const int width = (blockDim.x + 2 * RAD);
	const int s_idx = (threadIdx.y + RAD) * width + (threadIdx.x + RAD) ; 
	

	//Fill middle cells 
	s_in[s_idx] = d_x0[IX(i, j)];
	//Fill halo cells (boundary)
	if (threadIdx.x < RAD) {
		s_in[s_idx - RAD] = d_x0[i - RAD];
		s_in[s_idx + blockDim.x] = x_0[i + blockDim.x];
	}
	__syncthreads(); //Prevent race conditions

	// Diffuse operation (2D stencil)
	d_x[IX(i, j)] = (d_x0[IX(i, j)] + a * (s_in[IX(i - 1, j)] + s_in[IX(i + 1, j)] +
					s_in[IX(i, j - 1)] + s_in[IX(i, j + 1)])) / (1 + 4 * a);

}

void diffuse_parallel(std::vector<float>& x, std::vector<float>& x0, float diff, float dt, int N){
	float* x_new = &x[0];
	float* x_prev = &x0[0];

	float* d_x = 0;
	float* d_x0 = 0;

	//Allocate memory for on-device arrays
	cudaMalloc(&d_x, x.size() * sizeof(float));
	cudaMalloc(&d_x0, x0.size() * sizeof(float));
	//Copy contents from CPU memory to GPU memory
	cudaMemcpy(d_x0, x_prev, x0.size() * sizeof(float), cudaMemcpyHostToDevice);

	//Copy updated arryas from GPU back to CPU
	cudaMemcpy(x_new, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

	const size_t smemSize = (TPB + 2 * RAD) * (TPB + 2 * RAD) * sizeof(float); //Set size for shared memory array 
	diffuse_kernel<<<(n + TPB - 1)/TPB, TPB, smemSize>>>(d_x, d_x0, diff, dt, N);

	//Clear memory on GPU
	cudaFree(d_x);
	cudaFree(d_x0);
}