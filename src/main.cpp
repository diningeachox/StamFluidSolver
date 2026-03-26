#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <chrono>
#include <stdlib.h>     /* srand, rand */
#include <stdexcept>
#include <ctime>

#include <math.h>
#include "shader.h"
#include "density.h"


#define PI 3.14159265

int FPS = 30;
double MS_PER_UPDATE = 1000 / FPS;

static int win_id;
static int win_x, win_y;
static double omx, omy, mx, my;
static float force = 5.0f;

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
		//std::cout << "Right Cursor Pressed at (" << xpos << " : " << ypos << ")" << std::endl;
	}
	else if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
	{
		//getting cursor position
		glfwGetCursorPos(window, &xpos, &ypos);
		//std::cout << "Right Cursor Released at (" << xpos << " : " << ypos << ")" << std::endl;
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

		//std::cout << "Left Cursor Pressed at (" << mx << " : " << my << ")" << std::endl;

	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		//getting cursor position
		glfwGetCursorPos(window, &mx, &my);
		//std::cout << "Left Cursor Released at (" << mx << " : " << my << ")" << std::endl;
		//xpos = -1;
		//ypos = -1;
		mouse_down[0] = 0;
	}
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

int main()
{
    double lag = 0;

    int size = 900;
    int scale = 3; //How large to draw each grid

    //Simulation
    Simulation* sim = new Simulation(size / scale);
    int frame = 0;
    //Texture data
    std::vector<float> data;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //Ios
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(size, size, "StamFluidSolver", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    //Set dimensions of render window
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
		// Ensure we can capture the escape key being pressed below
		glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
		glfwSetMouseButtonCallback(window, mouse_button_callback);
		glfwSetKeyCallback(window, key_callback);
		glfwSetCursorPosCallback(window, mouse_move_callback);

    //Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //Initialize shader object using the Shader class
    Shader ourShader("textureshader.vs", "textureshader.fs");

    //Generate Texture
	unsigned int screen_tex;
    glGenTextures(1, &screen_tex);
    glBindTexture(GL_TEXTURE_2D, screen_tex);
    // set the texture wrapping/filtering options (on the currently bound texture object)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


	//Vertices of screen quad
	static const float vertices[] = {
			//Position           Texcoords
			1.0f,  -1.0f, 0.0f,  1.0f, 1.0f,  //Bottom right
			1.0f,  1.0f,  0.0f,  1.0f, 0.0f,  //Top right
			-1.0f, -1.0f, 0.0f,  0.0f, 1.0f,  //Bottom Left
			-1.0f, 1.0f,  0.0f,  0.0f, 0.0f,  //Top left
	};
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    //Create vertex buffer object and vertex array object
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO); //Vertex buffer
    glGenBuffers(1, &EBO); //Index buffer
    //Bind vertex array object
    glBindVertexArray(VAO);

    //Bind vertex buffer to GL_ARRAY_BUFFER
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    //Copy vertices array into vertex buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    //Bind index buffer and copy indices array into it
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    //Use shader program and draw
    ourShader.use();
    //Set texture units
    glUniform1i(glGetUniformLocation(ourShader.ID, "tex"), 0); // set it manually


    

    //Render loop
    while(!glfwWindowShouldClose(window))
    {
        //Handle input
        processInput(window);
				//Clear window
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

		/* ------------- Update simulation --------------*/
		sim->reset(size / scale);
		//std::cout << xpos << std::endl;
		//Add dye where mouse pressed
		if (xpos >= 0 && ypos >= 0) {
				int N = size / scale;

				int i = (int)((ypos / (float)size) * N + 1);
				int j = (int)((xpos / (float)size) * N + 1);
				//Add source
				sim->stamp(i, j, 1, 1, N, 500.0f * rgb[0], 100.0f * rgb[1], 100.0f * rgb[2]);
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

		ourShader.use();
		data = sim->getDensity(size / scale);

		



        /* ------------- Render --------------*/
		//Convert density arrays into a texture for the quad
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size / scale, size / scale, 0, GL_RGB, GL_FLOAT, data.data());

		//Position attribute
	    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	    glEnableVertexAttribArray(0);

	    //TextureCoord attribute
	    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	    glEnableVertexAttribArray(1);

        glBindVertexArray(VAO);

        //Draw quad
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        //Get events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    
    //Deallocation
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();

    

    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
