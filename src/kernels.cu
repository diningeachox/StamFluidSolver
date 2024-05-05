#include "kernels.cuh"


#define TPB 32 //threads per block
#define RAD 1 //radius of stencil

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void diffuse_kernel(float* d_x, float* d_x0, float diff, float dt, int N) {
	const float a = dt * diff * N * N;

	//Global index
	const int j = threadIdx.x + blockDim.x * blockIdx.x;
	const int i = threadIdx.y + blockDim.y * blockIdx.y;

	if (i > N || j > N || j < 1 || j < 1) return;

	//Local index 
	//Each shared memory block is (blockDim.x + 2) * (blockDim.y + 2)
	// Dynamically declared shared memory
	
	// extern __shared__ float s_in[];	
	__shared__ float s_in[TPB + 2][TPB + 2];

	//const int s_idx = threadIdx.x + RAD; 


	const int width = (blockDim.x + 2 * RAD);
	
	const int s_idx_x = threadIdx.x + RAD;
	const int s_idx_y = threadIdx.y + RAD;

	//Fill middle cells 
	s_in[s_idx_y][s_idx_x] = d_x[IX(i, j)];

	//Fill halo cells (boundary)

	if (threadIdx.x < RAD) {
		s_in[s_idx_y][s_idx_x - RAD] = d_x[IX(i, j - RAD)];
		s_in[s_idx_y][s_idx_x + blockDim.x] = d_x[IX(i, j + blockDim.x)];		
	}

	//__syncthreads(); //Prevent race conditions

	if (threadIdx.y < RAD) {
		s_in[s_idx_y - RAD][s_idx_x] = d_x[IX(i - RAD, j)];
		s_in[s_idx_y + blockDim.y][s_idx_x] = d_x[IX(i + blockDim.y, j)];		
	}

	__syncthreads(); //Prevent race conditions

	// Diffuse operation (2D stencil)
	// d_x[IX(i, j)] = (d_x0[IX(i, j)] + a * (s_in[IX(i - 1, j)] + s_in[IX(i + 1, j)] +
	// 				s_in[IX(i, j - 1)] + s_in[IX(i, j + 1)])) / (1 + 4 * a);

	d_x[IX(i, j)] = (d_x0[IX(i, j)] + a * (s_in[i - 1][j] + s_in[i + 1][j] +
					s_in[i][j - 1] + s_in[i][j + 1])) / (1 + 4 * a);

}

__global__ void diffuse_kernel_sm(float* d_out, float* d_in, float* d_x0, float diff, float dt, int N) {
	const float a = dt * diff * N * N;

	//Indexing function 
	auto idx = [&N](int y,int x){ return y*(N+2)+x; };

	// tiles overlap hence x0 & y0 strides reduced by twice halo width
	int x0 = (blockDim.x-2)*blockIdx.x; // x tile origin in array 
	int y0 = (blockDim.y-2)*blockIdx.y; // y tile origin in array
	int xa = x0+threadIdx.x;  // thread x in array       
	int ya = y0+threadIdx.y;  // thread y in array
	int xs = threadIdx.x;     // thread x in tile 
	int ys = threadIdx.y;     // thread y in tile
	if(xa >= N+2 || ya >= N+2) return; // out of range check

	//Each shared memory block is (blockDim.x + 2) * (blockDim.y + 2)
	// Dynamically declared shared memory
	__shared__ float s_in[TPB][TPB];

	//Copy entries to d_out
	d_out[idx(ya,xa)] = d_in[idx(ya,xa)];

	//Fill middle cells 
	s_in[ys][xs] = d_in[idx(ya, xa)];
	__syncthreads(); //Prevent race conditions

	if(xa < 1 || ya < 1 || xa >= N+1 || ya >= N+1) return; // inside array 
	if(xs < 1 || ys < 1 || xs >= TPB-1 || ys >= TPB-1) return; // inside tile 
	d_out[idx(ya,xa)] = (d_x0[idx(ya,xa)] + a * (s_in[ys - 1][xs] + s_in[ys + 1][xs] +
					s_in[ys][xs - 1] + s_in[ys][xs + 1])) / (1 + 4 * a);


}

__global__ void set_bnd_kernel(float* d_out, float* d_in, int b, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;  
	auto idx = [&N](int y,int x){ return y*(N+2)+x; };
	int xa = blockDim.x*blockIdx.x+threadIdx.x;  // thread x in array       
	int ya = blockDim.y*blockIdx.y+threadIdx.y;  // thread y in array

	//Copy entries to d_out
	d_out[idx(ya,xa)] = d_in[idx(ya,xa)];


	//Boundary conditions (edges)
	if (i >= 1 && i <= N) {
		d_out[idx(0, i)] = b == 1 ? -d_in[idx(1, i)] : d_in[idx(1, i)];
		d_out[idx(N + 1, i)] = b == 1 ? -d_in[idx(N, i)] : d_in[idx(N, i)];
		d_out[idx(i, 0)] = b == 2 ? -d_in[idx(i, 1)] : d_in[idx(i, 1)]; d_in[idx(i, N + 1)] = b == 2 ? -d_in[idx(i, N)] : d_in[idx(i, N)];
	}

	//Bounday conditions (corners) (only need to do this once)
	if (i == 0){
		d_out[idx(0, 0)] = 0.5 * (d_in[idx(1, 0)] + d_in[idx(0, 1)]);
		d_out[idx(0, N + 1)] = 0.5 * (d_in[idx(1, N + 1)] + d_in[idx(0, N)]);
		d_out[idx(N + 1, 0)] = 0.5 * (d_in[IX(N, 0)] + d_in[idx(N + 1, 1)]);
		d_out[idx(N + 1, N + 1)] = 0.5 * (d_in[idx(N, N + 1)] + d_in[idx(N + 1, N)]);
	}
}

void diffuse_parallel(std::vector<float>& x, std::vector<float>& x0, float diff, float dt, int b, int N){

	//Arrays from host
	float* x_new = x.data();
	float* x_prev = x0.data();

	//Arrays on device
	//float* d_x;
	float* d_x0;

	//These two arrays will alternately hold the latest updated values in the loop
	float* d_a;
	float* d_b;

	//Allocate memory for on-device arrays
	//gpuErrchk(cudaMalloc(&d_x, x.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_x0, x0.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_a, x.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_b, x.size() * sizeof(float)));

	//Copy contents from CPU memory to GPU memory
	gpuErrchk(cudaMemcpy(d_x0, x_prev, x0.size() * sizeof(float), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(d_x, x_new, x.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_a, x_new, x.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, x_new, x.size() * sizeof(float), cudaMemcpyHostToDevice));

	//Set kernel dimensions 
	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks((N + 2 + TPB - 1)/TPB, (N + 2 + TPB - 1)/TPB);

	//const size_t smemSize = (TPB + 2 * RAD) * (TPB + 2 * RAD) * sizeof(float); //Set size for shared memory array 
	//diffuse_kernel<<<numBlocks, threadsPerBlock>>>(d_x, d_x0, diff, dt, N);

	for (int k = 0; k < 20; k++){
		//diffuse_kernel_sm<<<numBlocks, threadsPerBlock>>>(d_x, d_x0, diff, dt, N);

		diffuse_kernel_sm<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_x0, diff, dt, N); //d_a holds updated values
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(d_b, d_a, b, N); //d_b holds updated values
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		
	}
	

	//Copy updated arryas from GPU back to CPU
	gpuErrchk(cudaMemcpy(x_new, d_a, x.size() * sizeof(float), cudaMemcpyDeviceToHost));

	//Copy back to vector 
	x.assign(x_new, x_new + x.size());

	//Clear memory on GPU
	//gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_x0));
	gpuErrchk(cudaFree(d_a));
	gpuErrchk(cudaFree(d_b));
}