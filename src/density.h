#pragma once
#include <iostream>
#include <vector>

//Macro for getting the index of a coordinate stored in 1D array
#define IX(i,j) ((i)+(N+2)*(j))

class Simulation{
private:
	int size; //Size of grid
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> x_prev;
	std::vector<float> y_prev;
	std::vector<float> dens;
	std::vector<float> dens_prev;

	std::vector<float> r;
	std::vector<float> r_prev;
	std::vector<float> g;
	std::vector<float> g_prev;
	std::vector<float> b;
	std::vector<float> b_prev;

	float viscosity = 0.0f; //Viscosity of the fluid
	float diffusion = 0.001f; //Diffusion rate

public:
	Simulation(int size);
	~Simulation();

	void add_source(int N, std::vector<float>& x, std::vector<float>& s, float dt);

	void diffuse(int N, int b, std::vector<float>& x, std::vector<float>& x0, float diff, float dt);

	void advect(int N, int b, std::vector<float>& d, std::vector<float>& d0, std::vector<float>& u, std::vector<float>& v, float dt);

	void set_bnd(int N, int b, std::vector<float>& x);

	//Update functions

	//Update density
	void dens_step(int N, std::vector<float>& x, std::vector<float>& x0, std::vector<float>& u, 
					std::vector<float>& v, float diff, float dt);

	//Update velocity
	void vel_step(int N, std::vector<float>& u, std::vector<float>& v, std::vector<float>& u0, std::vector<float>& v0, float visc, float dt);

	void project(int N, std::vector<float>& u, std::vector<float>& v, std::vector<float>& p, std::vector<float>& div);

	void update(int N, float dt);
	void reset(int N);


	std::vector<float> getDensity(int N);

	void stamp(int X, int Y, int W, int H, int N, float R, float G, float B);
	void add_force(int X, int Y, int N, float u, float v);
	void sink(int X, int Y, int W, int H, int N);
};