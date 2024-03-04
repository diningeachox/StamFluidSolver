#version 330 core

// Ouput data
//out vec4 color;
in vec3 Color;
in vec2 Texcoord;

out vec4 FragColor;

uniform sampler2D tex;

void main()
{

	// Output color = texture's color
	FragColor = texture(tex, Texcoord) * vec4(1.0, 1.0, 1.0, 1.0);

}
