#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 texcoord;

//out vec3 Color;
out vec2 Texcoord;

void main(){

    Texcoord = texcoord;
    gl_Position = vec4(vertexPosition_modelspace, 1.0);
}
