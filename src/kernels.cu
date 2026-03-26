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

__host__ __device__ void boundary_cond(float* d, int i, int N, int b, bool corner){
	auto idx = [&N](int x,int y){ return y*(N+2)+x; };

	//Boundary conditions (edges)
	if (!corner){
		if (i <= N) {
			d[idx(0, i)] = b == 1 ? -d[idx(1, i)] : d[idx(1, i)];
			d[idx(N + 1, i)] = b == 1 ? -d[idx(N, i)] : d[idx(N, i)];
			d[idx(i, 0)] = b == 2 ? -d[idx(i, 1)] : d[idx(i, 1)]; 
			d[idx(i, N + 1)] = b == 2 ? -d[idx(i, N)] : d[idx(i, N)];

		}
	}
	//Bounday conditions (corners) (only need to do this once)
	else { // Only xa=ya=0
		d[idx(0, 0)] = 0.5 * (d[idx(1, 0)] + d[idx(0, 1)]);
		d[idx(0, N + 1)] = 0.5 * (d[idx(1, N + 1)] + d[idx(0, N)]);
		d[idx(N + 1, 0)] = 0.5 * (d[idx(N, 0)] + d[idx(N + 1, 1)]);
		d[idx(N + 1, N + 1)] = 0.5 * (d[idx(N, N + 1)] + d[idx(N + 1, N)]);
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

	__syncthreads(); //Prevent race conditions

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
	auto idx = [&N](int x,int y){ return y*(N+2)+x; };

	// tiles overlap hence x0 & y0 strides reduced by twice halo width
	//int x0 = (blockDim.x-2)*blockIdx.x; // x tile origin in array 
	//int y0 = (blockDim.y-2)*blockIdx.y; // y tile origin in array
	int xa = blockDim.x*blockIdx.x+threadIdx.x;  // thread x in array       
	int ya = blockDim.y*blockIdx.y+threadIdx.y;  // thread y in array

	const int xs = threadIdx.x + RAD;
	const int ys = threadIdx.y + RAD;

	//Each shared memory block is (blockDim.x + 2) * (blockDim.y + 2)
	// Dynamically declared shared memory
	__shared__ float s_in[TPB+2][TPB+2];

	//Copy entries to d_out
	//d_out[idx(ya,xa)] = d_in[idx(ya,xa)];

	//Fill middle cells 
	s_in[ys][xs] = d_in[idx(xa, ya)];
	//__syncthreads(); //Prevent race conditions

	//Fill halo cells (boundary)
	if (threadIdx.x < RAD) {
		s_in[ys][xs - RAD] = d_in[idx(xa - RAD, ya)];
		s_in[ys][xs + blockDim.x] = d_in[idx(xa + blockDim.x, ya)];		
	}

	if (threadIdx.y < RAD) {
		s_in[ys - RAD][xs] = d_in[idx(xa, ya - RAD)];
		s_in[ys + blockDim.y][xs] = d_in[idx(xa, ya + blockDim.y)];		
	}

	if (threadIdx.x < RAD && threadIdx.y < RAD){
		s_in[ys - RAD][xs - RAD] = 0.0f;
	} 

	__syncthreads(); 
	
	if(xs < RAD || ys < RAD || xs >= RAD + blockDim.x || ys >= RAD + blockDim.y) return; // outside array 
	if(xa < 1 || ya < 1 || xa >= N+1 || ya >= N+1) return; // outside array 
	d_out[idx(xa, ya)] = (d_x0[idx(xa, ya)] + a * (s_in[ys - 1][xs] + s_in[ys + 1][xs] +
					s_in[ys][xs - 1] + s_in[ys][xs + 1])) / (1 + 4 * a);
}

/**
 * Uses overlapped tiling to reduce read/writes to global memory
 * Shared memory = (32+20*2)^2 * 4 = 20.736 KB
 * Enough to fit into the 48 KB limit for share memory blocks
 */
__global__ void diffuse_kernel_overlapped_tiling(float* d_out, float* d_in, float* d_x0, float diff, float dt, float a, float b, int N, int T) {
	//Indexing function 
	auto idx = [&N](int x,int y){ return y*(N+2)+x; };

	// tiles overlap hence x0 & y0 strides reduced by twice halo width

	int xa = blockDim.x*blockIdx.x+threadIdx.x;  // thread x in array       
	int ya = blockDim.y*blockIdx.y+threadIdx.y;  // thread y in array

	const int xs = threadIdx.x + T;
	const int ys = threadIdx.y + T;

	// Shared memory (used for double buffering)
	__shared__ float s_in[2][TPB+40][TPB+40]; //T=20 -> 72x72x2 ~ 42 KB for each thread block

	//Copy entries to d_out
	//d_out[idx(ya,xa)] = d_in[idx(ya,xa)];

	//Fill middle cells 
	s_in[0][ys][xs] = d_in[idx(xa, ya)];
	//__syncthreads(); //Prevent race conditions

	//Fill halo cells (boundary)
	if (threadIdx.x < T) {
		s_in[0][ys][xs - T] = d_in[idx(xa - T, ya)];
		s_in[0][ys][xs + blockDim.x] = d_in[idx(xa + blockDim.x, ya)];		
	}

	if (threadIdx.y < RAD) {
		s_in[0][ys - T][xs] = d_in[idx(xa, ya - T)];
		s_in[0][ys + blockDim.y][xs] = d_in[idx(xa, ya + blockDim.y)];		
	}

	// if (threadIdx.x < T && threadIdx.y < T){
	// 	s_in[ys - T][xs - T] = 0.0f;
	// } 

	//Run through iterations
	for (int i = 0; i < T; i++){

		s_in[(i+1)%2][ys][xs] = (s_in[i%2][ys][xs] + a * (s_in[i%2][ys - 1][xs] + s_in[i%2][ys + 1][xs] +
					s_in[i%2][ys][xs - 1] + s_in[i%2][ys][xs + 1])) / (1 + 4 * a);

		//set_bnd logic
		// if (ya == 0 && xa >= 1 && xa <= N){
		// 	__syncthreads();
		// 	boundary_cond(s_in[(i+1)%2], xa, N, b, false);
		// } else if (xa == 0){
		// 	__syncthreads();
		// 	boundary_cond(s_in[(i+1)%2], xa, N, b, true);
		// }
		__syncthreads();
	}

	 
	
	if(xs < T || ys < T|| xs >= T + blockDim.x || ys >= T + blockDim.y) return; // outside array 
	if(xa < 1 || ya < 1 || xa >= N+1 || ya >= N+1) return; // outside array 

	//Write back to global memory
	d_out[idx(xa, ya)] = s_in[T%2][ys][xs];

}


__global__ void set_bnd_kernel(float* d_out, float* d_in, int b, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;  
	auto idx = [&N](int x,int y){ return y*(N+2)+x; };
	int xa = blockDim.x*blockIdx.x+threadIdx.x;  // thread x in array       
	int ya = blockDim.y*blockIdx.y+threadIdx.y;  // thread y in array
	if(xa >= N+2 || ya >= N+2) return; 

	//Copy entries to d_out
	d_out[idx(xa, ya)] = d_in[idx(xa, ya)];

	//Boundary conditions (edges)
	if (i >= 1 && i <= N && ya == 0) {

		d_out[idx(0, i)] = b == 1 ? -d_in[idx(1, i)] : d_in[idx(1, i)];

		d_out[idx(N + 1, i)] = b == 1 ? -d_in[idx(N, i)] : d_in[idx(N, i)];
		d_out[idx(i, 0)] = b == 2 ? -d_in[idx(i, 1)] : d_in[idx(i, 1)]; 
		d_out[idx(i, N + 1)] = b == 2 ? -d_in[idx(i, N)] : d_in[idx(i, N)];


	}

	//Bounday conditions (corners) (only need to do this once)
	if (xa == 0 && ya == 0){
		d_out[idx(0, 0)] = 0.5 * (d_in[idx(1, 0)] + d_in[idx(0, 1)]);
		d_out[idx(0, N + 1)] = 0.5 * (d_in[idx(1, N + 1)] + d_in[idx(0, N)]);
		d_out[idx(N + 1, 0)] = 0.5 * (d_in[idx(N, 0)] + d_in[idx(N + 1, 1)]);
		d_out[idx(N + 1, N + 1)] = 0.5 * (d_in[idx(N, N + 1)] + d_in[idx(N + 1, N)]);
	}
}

__global__ void advect_kernel(float* d_out, float* d_in, float* d_u, float* d_v, int b, float dt, int N){
	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;
	dt0 = dt * N;

	auto idx = [&N](int x,int y){ return y*(N+2)+x; };
	int xa = blockDim.x*blockIdx.x+threadIdx.x;  // thread x in array       
	int ya = blockDim.y*blockIdx.y+threadIdx.y;  // thread y in array
	//if(xa >= N+2 || ya >= N+2) return; 

	if(xa < 1 || ya < 1 || xa >= N+1 || ya >= N+1) return; // boundary conditions 

	// Dynamically declared shared memory
	//__shared__ float s_in[TPB+2][TPB+2];


	// Update with the value of the field at the destination of the velocity vector at source
	// The value of the field is interpolated by nearest cells (bilinearly)
	x = xa - dt0 * d_u[idx(xa, ya)]; 
	y = ya - dt0 * d_v[idx(xa, ya)];
	if (x < 0.5) x = 0.5; if (x > N + 0.5) x = N + 0.5; i0 = (int)x; i1 = i0 + 1;
	if (y < 0.5) y = 0.5; if (y > N + 0.5) y = N + 0.5; j0 = (int)y; j1 = j0 + 1;
	s1 = x - i0; s0 = 1 - s1; t1 = y - j0; t0 = 1 - t1;
	d_out[idx(xa, ya)] = s0 * (t0 * d_in[idx(i0, j0)] + t1 * d_in[idx(i0, j1)]) +
		s1 * (t0 * d_in[idx(i1, j0)] + t1 * d_in[idx(i1, j1)]);
	
}

__global__ void project_kernel(float* d_u, float* d_v, float* d_p, float* d_div, int N) {
	

	//Indexing function 
	auto idx = [&N](int x,int y){ return y*(N+2)+x; };

	// tiles overlap hence x0 & y0 strides reduced by twice halo width
	//int x0 = (blockDim.x-2)*blockIdx.x; // x tile origin in array 
	//int y0 = (blockDim.y-2)*blockIdx.y; // y tile origin in array
	int xa = blockDim.x*blockIdx.x+threadIdx.x;  // thread x in array       
	int ya = blockDim.y*blockIdx.y+threadIdx.y;  // thread y in array

	const int xs = threadIdx.x + RAD;
	const int ys = threadIdx.y + RAD;

	//Each shared memory block is (blockDim.x + 2) * (blockDim.y + 2)
	// Dynamically declared shared memory
	__shared__ float s_p[TPB+2][TPB+2];

	//Copy entries to d_out
	//d_out[idx(ya,xa)] = d_in[idx(ya,xa)];

	//Fill middle cells 
	s_p[ys][xs] = d_p[idx(xa, ya)];
	//__syncthreads(); //Prevent race conditions

	//Fill halo cells (boundary)
	if (threadIdx.x < RAD) {
		s_p[ys][xs - RAD] = d_p[idx(xa - RAD, ya)];
		s_p[ys][xs + blockDim.x] = d_p[idx(xa + blockDim.x, ya)];		
	}

	if (threadIdx.y < RAD) {
		s_p[ys - RAD][xs] = d_p[idx(xa, ya - RAD)];
		s_p[ys + blockDim.y][xs] = d_p[idx(xa, ya + blockDim.y)];		
	}

	if (threadIdx.x < RAD && threadIdx.y < RAD){
		s_p[ys - RAD][xs - RAD] = 0.0f;
	} 

	__syncthreads(); 
	
	if(xa < 1 || ya < 1 || xa >= N+1 || ya >= N+1) return; // outside array 
	d_p[idx(xa, ya)] = (d_div[idx(xa, ya)] + s_p[ys - 1][xs] + s_p[ys + 1][xs] +
					s_p[ys][xs - 1] + s_p[ys][xs + 1]) / 4.0f;


}

/**
 * Uses overlapped tiling to reduce read/writes to global memory
 * Shared memory = (32+20*2)^2 * 4 = 20.736 KB
 * Enough to fit into the 48 KB limit for share memory blocks
 */
__global__ void project_kernel_overlapped_tiling(float* d_u, float* d_v, float* d_p, float* d_div, int N, int T) {

	float h = 1.0f / N;

	//Indexing function 
	auto idx = [&N](int x,int y){ return y*(N+2)+x; };

	// tiles overlap hence x0 & y0 strides reduced by twice halo width

	int xa = blockDim.x*blockIdx.x+threadIdx.x;  // thread x in array       
	int ya = blockDim.y*blockIdx.y+threadIdx.y;  // thread y in array

	const int xs = threadIdx.x + T;
	const int ys = threadIdx.y + T;

	/*
	* Update d_div (only 1 iteration)
	*/
	d_div[idx(xa, ya)] = -0.5 * h * (d_u[idx(xa+1, ya)] - d_u[idx(xa-1, ya)] +
				d_v[idx(xa, ya+1)] - d_v[idx(xa, ya-1)]);
	d_p[idx(xa, ya)] = 0;

	// TODO: 
	// set_bnd(N, 0, div); set_bnd(N, 0, p);

	// Shared memory (used for double buffering)
	__shared__ float s_in[2][TPB+40][TPB+40]; //T=20 -> 72x72x2 ~ 42 KB for each thread block

	/*
	* Update d_p (T=20 iterations)
	*/
	
	s_in[0][ys][xs] = d_p[idx(xa, ya)]; //Fill middle cells 
	//__syncthreads(); //Prevent race conditions

	//Fill halo cells (boundary)
	if (threadIdx.x < T) {
		s_in[0][ys][xs - T] = d_p[idx(xa - T, ya)];
		s_in[0][ys][xs + blockDim.x] = d_p[idx(xa + blockDim.x, ya)];		
	}

	if (threadIdx.y < RAD) {
		s_in[0][ys - T][xs] = d_p[idx(xa, ya - T)];
		s_in[0][ys + blockDim.y][xs] = d_p[idx(xa, ya + blockDim.y)];		
	}

	//Run through iterations
	for (int i = 0; i < T; i++){

		s_in[(i+1)%2][ys][xs] = (d_div[idx(xa, ya)] + s_in[i%2][ys - 1][xs] + s_in[i%2][ys + 1][xs] +
					s_in[i%2][ys][xs - 1] + s_in[i%2][ys][xs + 1]) / 4;

		//set_bnd logic

		// if (ya == 0 && xa >= 1 && xa <= N){
		// 	__syncthreads();
		// 	boundary_cond(s_in[(i+1)%2], xa, N, b, false);
		// } else if (xa == 0){
		// 	__syncthreads();
		// 	boundary_cond(s_in[(i+1)%2], xa, N, b, true);
		// }
		__syncthreads();
	}

	//Write back to global memory
	d_p[idx(xa, ya)] = s_in[T%2][ys][xs];

	/*
	* Update d_u, d_v (only 1 iteration)
	*/
	d_u[idx(xa, ya)] -= 0.5 * (d_p[idx(xa+1, ya)] - d_p[idx(xa-1, ya)]) / h;
	d_v[idx(xa, ya)] -= 0.5 * (d_p[idx(xa, ya+1)] - d_p[idx(xa, ya-1)]) / h;

	// set_bnd(N, 1, u); 
	// set_bnd(N, 2, v);

	
	if(xs < T || ys < T|| xs >= T + blockDim.x || ys >= T + blockDim.y) return; // outside array 
	if(xa < 1 || ya < 1 || xa >= N+1 || ya >= N+1) return; // outside array 

	

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

	//Allocate CPU arrays
	// gpuErrchk(cudaMallocHost(&x_prev, x0.size() * sizeof(float)));
	// gpuErrchk(cudaMallocHost(&x_new, x.size() * sizeof(float)));

	// x_new = x.data();
	// x_prev = x0.data();


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

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); // Record start event in stream 0 (default)
	for (int k = 0; k < 20; k++){
		diffuse_kernel_sm<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_x0, diff, dt, N); //d_a holds updated values
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(d_b, d_a, b, N); //d_b holds updated values
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		
	}

	cudaEventRecord(stop, 0);  // Record stop event in stream 0

	cudaEventSynchronize(stop); // Wait for the stop event to complete
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop); // Calculate elapsed time

	std::cout << "Diffuse kernel took: " << milliseconds << " ms" << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	

	//Copy updated arryas from GPU back to CPU
	gpuErrchk(cudaMemcpy(x_new, d_b, x.size() * sizeof(float), cudaMemcpyDeviceToHost));

	//Copy back to vector 
	x.assign(x_new, x_new + x.size());

	//Clear memory on CPU
	// gpuErrchk(cudaFreeHost(x_prev));
	// gpuErrchk(cudaFreeHost(x_new));
	// free(x_prev);
	// free(x_new);

	//Clear memory on GPU
	//gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_x0));
	gpuErrchk(cudaFree(d_a));
	gpuErrchk(cudaFree(d_b));
}

void advect_parallel(std::vector<float>& d, std::vector<float>& d0, std::vector<float>& u, std::vector<float>& v, float dt, int b, int N){

	//Arrays from host
	float* x_new = d.data();
	float* x_prev = d0.data();
	float* x_u = u.data();
	float* x_v = v.data();

	//Arrays on device
	//float* d_x;
	float* d_x0;
	float* d_out;

	float* d_u;
	float* d_v;

	//Allocate memory for on-device arrays
	//gpuErrchk(cudaMalloc(&d_x, x.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_x0, d0.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_out, d.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_u, u.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_v, v.size() * sizeof(float)));

	//Copy contents from CPU memory to GPU memory
	gpuErrchk(cudaMemcpy(d_x0, x_prev, d0.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_out, x_new, d.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_u, x_u, u.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_v, x_v, v.size() * sizeof(float), cudaMemcpyHostToDevice));

	//Set kernel dimensions 
	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks((N + 2 + TPB - 1)/TPB, (N + 2 + TPB - 1)/TPB);


	advect_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_x0, d_u, d_v, b, dt, N); //d_a holds updated values
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(d_x0, d_out, b, N); //d_b holds updated values
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
		
	
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//Copy updated arryas from GPU back to CPU
	gpuErrchk(cudaMemcpy(x_new, d_x0, d.size() * sizeof(float), cudaMemcpyDeviceToHost));

	//Copy back to vector 
	d.assign(x_new, x_new + d.size());

	//Clear memory on GPU
	//gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_x0));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_u));
	gpuErrchk(cudaFree(d_v));
}

void project_parallel(std::vector<float>& d, std::vector<float>& d0, std::vector<float>& u, std::vector<float>& v, float dt, int b, int N){

	//Arrays from host
	float* x_new = d.data();
	float* x_prev = d0.data();
	float* x_u = u.data();
	float* x_v = v.data();

	//Arrays on device
	//float* d_x;
	float* d_x0;
	float* d_out;

	float* d_u;
	float* d_v;

	//Allocate memory for on-device arrays
	//gpuErrchk(cudaMalloc(&d_x, x.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_x0, d0.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_out, d.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_u, u.size() * sizeof(float)));
	gpuErrchk(cudaMalloc(&d_v, v.size() * sizeof(float)));

	//Copy contents from CPU memory to GPU memory
	gpuErrchk(cudaMemcpy(d_x0, x_prev, d0.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_out, x_new, d.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_u, x_new, u.size() * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_v, x_new, v.size() * sizeof(float), cudaMemcpyHostToDevice));

	//Set kernel dimensions 
	dim3 threadsPerBlock(TPB, TPB);
	dim3 numBlocks((N + 2 + TPB - 1)/TPB, (N + 2 + TPB - 1)/TPB);

	//const size_t smemSize = (TPB + 2 * RAD) * (TPB + 2 * RAD) * sizeof(float); //Set size for shared memory array 
	//diffuse_kernel<<<numBlocks, threadsPerBlock>>>(d_x, d_x0, diff, dt, N);

	//diffuse_kernel_sm<<<numBlocks, threadsPerBlock>>>(d_x, d_x0, diff, dt, N);

	advect_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_x0, d_u, d_v, b, dt, N); //d_a holds updated values
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
	set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(d_x0, d_out, b, N); //d_b holds updated values
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );
		
	
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//Copy updated arryas from GPU back to CPU
	gpuErrchk(cudaMemcpy(x_new, d_x0, d.size() * sizeof(float), cudaMemcpyDeviceToHost));

	//Copy back to vector 
	d.assign(x_new, x_new + d.size());

	//Clear memory on GPU
	//gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_x0));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_u));
	gpuErrchk(cudaFree(d_v));
}