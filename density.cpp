#include "density.h"

Simulation::Simulation(int n) {
	size = (n+2)*(n+2);
	//Fill the vectors with 0
	x.resize(size);
	std::fill(x.begin(), x.end(), 0.0);

	y.resize(size);
	std::fill(y.begin(), y.end(), 0.0);

	x_prev.resize(size);
	std::fill(x_prev.begin(), x_prev.end(), 2.0);

	y_prev.resize(size);
	std::fill(y_prev.begin(), y_prev.end(), -2.0);

	dens.resize(size);
	std::fill(dens.begin(), dens.end(), 0.0);

	dens_prev.resize(size);
	std::fill(dens_prev.begin(), dens_prev.end(), 0.0);

	r.resize(size);
	std::fill(r.begin(), r.end(), 0.0);

	r_prev.resize(size);
	std::fill(r_prev.begin(), r_prev.end(), 0.0);

	g.resize(size);
	std::fill(g.begin(), g.end(), 0.0);

	g_prev.resize(size);
	std::fill(g_prev.begin(), g_prev.end(), 0.0);

	b.resize(size);
	std::fill(b.begin(), b.end(), 0.0);

	b_prev.resize(size);
	std::fill(b_prev.begin(), b_prev.end(), 0.0);
}

Simulation::~Simulation()
{
}

void Simulation::add_source(int N, std::vector<float>& x, std::vector<float>& s, float dt)
{
	for (int i = 0; i < size; i++) {
		x[i] += dt * s[i];
	}
}

void Simulation::diffuse(int N, int b, std::vector<float>& x, std::vector<float>& x0, float diff, float dt)
{
	int i, j, k;
	float a = dt * diff * N * N;
	for (k = 0; k < 20; k++) {
		for (i = 1; i <= N; i++) {
			for (j = 1; j <= N; j++) {
				x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
					x[IX(i, j - 1)] + x[IX(i, j + 1)])) / (1 + 4 * a);
			}
		}
		set_bnd(N, b, x);
	}
}

void Simulation::advect(int N, int b, std::vector<float>& d, std::vector<float>& d0, std::vector<float>& u, std::vector<float>& v, float dt)
{
	int i, j, i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;
	dt0 = dt * N;
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			x = i - dt0 * u[IX(i, j)]; y = j - dt0 * v[IX(i, j)];
			if (x < 0.5) x = 0.5; if (x > N + 0.5) x = N + 0.5; i0 = (int)x; i1 = i0 + 1;
			if (y < 0.5) y = 0.5; if (y > N + 0.5) y = N + 0.5; j0 = (int)y; j1 = j0 + 1;
			s1 = x - i0; s0 = 1 - s1; t1 = y - j0; t0 = 1 - t1;
			d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
				s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
		}
	}
	set_bnd(N, b, d);
}

void Simulation::set_bnd(int N, int b, std::vector<float>& x)
{
	for (int i = 1; i <= N; i++) {
		x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
		x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
		x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)]; x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
	}
	x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
	x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
	x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
	x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

void Simulation::dens_step(int N, std::vector<float>& x, std::vector<float>& x0, std::vector<float>& u, std::vector<float>& v, float diff, float dt)
{
	add_source(N, x, x0, dt);
	std::swap(x, x0); //Swap the two vectors
	diffuse(N, 0, x, x0, diff, dt);
	std::swap(x, x0);
	advect(N, 0, x, x0, u, v, dt);
}

void Simulation::vel_step(int N, std::vector<float>& u, std::vector<float>& v, std::vector<float>& u0, std::vector<float>& v0, float visc, float dt)
{
	add_source(N, u, u0, dt); 
	add_source(N, v, v0, dt);

	std::swap(u0, u); 
	diffuse(N, 1, u, u0, visc, dt);

	std::swap(v0, v);
	diffuse(N, 2, v, v0, visc, dt);

	project(N, u, v, u0, v0);

	std::swap(u0, u);
	std::swap(v0, v);

	advect(N, 1, u, u0, u0, v0, dt); 
	advect(N, 2, v, v0, u0, v0, dt);

	project(N, u, v, u0, v0);
}

void Simulation::project(int N, std::vector<float>& u, std::vector<float>& v, std::vector<float>& p, std::vector<float>& div)
{
	int i, j, k;
	float h;
	h = 1.0 / N;
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
				v[IX(i, j + 1)] - v[IX(i, j - 1)]);
			p[IX(i, j)] = 0;
		}
	}
	set_bnd(N, 0, div); set_bnd(N, 0, p);
	for (k = 0; k < 20; k++) {
		for (i = 1; i <= N; i++) {
			for (j = 1; j <= N; j++) {
				p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
					p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
			}
		}
		set_bnd(N, 0, p);
	}
	for (i = 1; i <= N; i++) {
		for (j = 1; j <= N; j++) {
			u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
			v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
		}
	}
	set_bnd(N, 1, u); 
	set_bnd(N, 2, v);
}

void Simulation::update(int N, float dt)
{
	


	vel_step(N, x, y, x_prev, y_prev, viscosity, dt);
	//dens_step(N, dens, dens_prev, x, y, diffusion, dt);
	dens_step(N, r, r_prev, x, y, diffusion, dt);
	dens_step(N, g, g_prev, x, y, diffusion, dt);
	dens_step(N, b, b_prev, x, y, diffusion, dt);


	
}

std::vector<float> Simulation::getDensity(int N)
{
	std::vector<float> amal;
	//Amalgamate the data in the r, g, b vectors into a pixel array
	int i, j;
	float total = 0;
	for (int k = 0; k < size; k++) {
		i = k / (N + 2);
		j = k % (N + 2);
		//If not on boundary push into amal
		if (i != 0 && i != N + 1 && j != 0 && j != N + 1) {
			//amal.push_back(std::min(r[k], 1.0f));
			//amal.push_back(std::min(g[k], 1.0f));
			//amal.push_back(std::min(b[k], 1.0f));

			amal.push_back(std::max(r[k], 0.0f));
			amal.push_back(g[k]);
			amal.push_back(b[k]);
			//total += r[k] + g[k] + b[k];
		}
	}
	//std::cout << total << std::endl;

	return amal;
}

void Simulation::stamp(int X, int Y, int W, int H, int N, float R, float G, float B)
{
	//Throw exception if rectangular stamp is out of range
	if (X <= 0 || X > N || Y <= 0 || Y > N || X + W > N || Y + H > N) {
		throw std::out_of_range("Index out of range.");
	}

	for (int i = Y; i < Y + H; i++) {
		for (int j = X; j < X + W; j++) {
			//r[IX(i, j)] = R;
			//g[IX(i, j)] = G;
			//b[IX(i, j)] = B;
			r_prev[IX(i, j)] += R;
			g_prev[IX(i, j)] += G;
			b_prev[IX(i, j)] += B;
		}
	}
}

void Simulation::sink(int X, int Y, int W, int H, int N) {
	if (X <= 0 || X > N || Y <= 0 || Y > N || X + W > N || Y + H > N) {
		throw std::out_of_range("Index out of range.");
	}

	for (int i = Y; i < Y + H; i++) {
		for (int j = X; j < X + W; j++) {
			//r[IX(i, j)] = R;
			//g[IX(i, j)] = G;
			//b[IX(i, j)] = B;
			r[IX(i, j)] = 0.0f;
			g[IX(i, j)] = 0.0f;
			b[IX(i, j)] = 0.0f;
			//x[IX(i, j)] = 0.0f;
			//y[IX(i, j)] = 0.0f;
		}
	}
}

void Simulation::add_force(int X, int Y, int N, float u, float v)
{
	if (X <= 0 || X > N || Y <= 0 || Y > N) {
		throw std::out_of_range("Index out of range.");
	}
	x_prev[IX(Y, X)] = u;
	y_prev[IX(Y, X)] = v;

}

void Simulation::reset(int N) {
	std::fill(x_prev.begin(), x_prev.end(), 0.0);
	std::fill(y_prev.begin(), y_prev.end(), 0.0);
	std::fill(r_prev.begin(), r_prev.end(), 0.0);
	std::fill(g_prev.begin(), g_prev.end(), 0.0);
	std::fill(b_prev.begin(), b_prev.end(), 0.0);

	//Sinks
	//sink(10, 10, 2, 2, N);
	//sink(20, 20, 2, 2, N);
}
