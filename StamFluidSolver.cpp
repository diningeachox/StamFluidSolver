// StamFluidSolver.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include <stdlib.h>     /* srand, rand */
#include <stdexcept>
#include <ctime>


// Include GLEW. Always include it before gl.h and glfw3.h, since it's a bit magic.
#include <GL/glew.h>
// Include GLFW
#include <GLFW/glfw3.h>

#include <GL/glut.h>

// Include GLM
#include <glm/glm.hpp>
#include "density.h"
#include "shader.hpp"

using namespace glm;

GLFWwindow* window;

typedef std::chrono::steady_clock Clock;

int FPS = 30;
double MS_PER_UPDATE = 1000 / FPS;

static int win_id;
static int win_x, win_y;
static double omx, omy, mx, my;
static float force = 5.0f;;

static int mouse_down[2] = { 0, 0 };


static double xpos = -1, ypos = -1;
static float rgb[3] = { 1.0f, 0.0f, 0.0f };
//Key Events
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_Q && action == GLFW_PRESS) { //Red
		rgb[0] = 1.0f;
		rgb[1] = 0.0f;
		rgb[2] = 0.0f;
		std::cout << "Switching to red" << std::endl;
	} 
	else if (key == GLFW_KEY_W && action == GLFW_PRESS) { //Green
		rgb[0] = 0.0f;
		rgb[1] = 1.0f;
		rgb[2] = 0.0f;
		std::cout << "Switching to green" << std::endl;
	}
	else if (key == GLFW_KEY_E && action == GLFW_PRESS) { //Blue
		rgb[0] = 0.0f;
		rgb[1] = 0.0f;
		rgb[2] = 1.0f;
		std::cout << "Switching to blue" << std::endl;
	}
		
}

//Mouse events

void mouse_move_callback(GLFWwindow* window, double x, double y)
{
	mx = x;
	my = y;
	//std::cout << "cursor moved:   " << x << " | " << y << std::endl;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
		//getting cursor position
		glfwGetCursorPos(window, &xpos, &ypos);
		std::cout << "Right Cursor Pressed at (" << xpos << " : " << ypos << ")" << std::endl;
	}
	else if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
	{
		//getting cursor position
		glfwGetCursorPos(window, &xpos, &ypos);
		std::cout << "Right Cursor Released at (" << xpos << " : " << ypos << ")" << std::endl;
		xpos = -1;
		ypos = -1;

	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		//getting cursor position
		glfwGetCursorPos(window, &mx, &my);

		//Record original position of the press
		omx = mx;
		omy = my;
		
		
		mouse_down[0] = 1;

		std::cout << "Left Cursor Pressed at (" << mx << " : " << my << ")" << std::endl;
		
	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		//getting cursor position
		glfwGetCursorPos(window, &mx, &my);
		std::cout << "Left Cursor Released at (" << mx << " : " << my << ")" << std::endl;
		//xpos = -1;
		//ypos = -1;
		mouse_down[0] = 0;
	}
}


int main()
{	
	//Variables for game loop
	
	double lag = 0;

	int size = 720;
	int scale = 12; //How large to draw each grid

	//Simulation 
	Simulation* sim = new Simulation(size / scale);
	int frame = 0;
	std::vector<float> data;

	//Add initial sources 
	//sim->stamp(8, 12, 4, 4, size / scale, 100.0f, 50.0f, 0.0f);
	//sim->stamp(20, 20, 5, 5, size / scale, 0.0f, 30.0f, 100.0f);

	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		//getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(size, size, "StamFluidSolver", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");		
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouse_move_callback);

	/*
	* Actual displays
	*/

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.8f, 0.0f);

	
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders("TextureVertexShader.vertexshader", "TextureFragmentShader.fragmentshader");

	
	//Vertices of rectangle
	static const GLfloat g_vertex_buffer_data[] = { 
		//Position           Texcoords
		1.0f,  -1.0f, 0.0f,  1.0f, 1.0f,  //Bottom right
		1.0f,  1.0f,  0.0f,  1.0f, 0.0f,  //Top right
		-1.0f, -1.0f, 0.0f,  0.0f, 1.0f,  //Bottom Left
		-1.0f, 1.0f,  0.0f,  0.0f, 0.0f,  //Top left
	};

	//Create buffers
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);


	//Checkerboard texture
	float pixels[] = {
	0.0f, 0.0f, 0.0f,   1.0f, 0.5f, 0.5f,
	1.0f, 1.0f, 1.0f,   0.0f, 0.0f, 0.0f
	};
	

	//Create 2D texture
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	//Load pixel data into 2D texture
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_FLOAT, pixels);
	//Interpolation
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glUniform1i(glGetUniformLocation(programID, "tex"), 0);

	
	clock_t prev = clock();
	do {
		// Clear the screen. It's not mentioned before Tutorial 02, but it can cause flickering, so it's there nonetheless.
		glClear(GL_COLOR_BUFFER_BIT);

		//Updates for flow
		clock_t current = clock();
		float elapsed = current - prev;
		prev = current;
		lag += elapsed;
		//r->ListenToEvents(); //Handle events


		while (lag >= MS_PER_UPDATE) {
			lag -= MS_PER_UPDATE;
			//std::cout << "Lag: " << lag << std::endl;

			sim->reset(size / scale);
			//Add dye where mouse pressed 
			if (xpos >= 0 && ypos >= 0) {
				int N = size / scale;

				int i = (int)((ypos / (float)size) * N + 1);
				int j = (int)((xpos / (float)size) * N + 1);
				//Add source
				sim->stamp(i, j, 1, 1, N, 100.0f * rgb[0], 100.0f * rgb[1], 100.0f * rgb[2]);
				//Add velocity vectors 

			}
			//std::cout << mx << "," << omx << std::endl;

			if (mouse_down[0]) {
				int N = size / scale;
				if (my >= size) my = size - 1;
				if (mx >= size) mx = size - 1;
				if (my < 0) my = 0;
				if (mx < 0) mx = 0;
				int i = (int)((my / (float)size) * N + 1);
				int j = (int)((mx / (float)size) * N + 1);
				sim->add_force(i, j, N, force * (mx - omx), force * (my - omy));
				//std::cout << mx << "," << omx << std::endl;

				omx = mx;
				omy = my;
				
			}


			sim->update(size / scale, 0.1f);
		}

		// Drawing
		//Draw rectangle test

		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size / scale, size / scale, 0, GL_RGBA, GL_FLOAT, data);

		// Use our shader
		glUseProgram(programID);

		data = sim->getDensity(size / scale);

		//Convert density arrays into a texture for the quad
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size / scale, size / scale, 0, GL_RGB, GL_FLOAT, data.data());

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			5 * sizeof(float),  // stride
			(void*)0            // array buffer offset
		);

		//2nd attribute buffer: texture attribute
		GLint texAttrib = glGetAttribLocation(programID, "texcoord");
		glEnableVertexAttribArray(texAttrib);
		glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE,
			5 * sizeof(float), (void*)(3 * sizeof(float)));

		// Draw the quad !
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); //4 vertices

		// Draw a rectangle from the 2 triangles using 6 indices
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(texAttrib);
		
		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
		frame++;

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();


	return 0;


}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
