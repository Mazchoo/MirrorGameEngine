#version 330 core

layout (location=0) in vec3 vertexPos;
layout (location=1) in vec2 vertexTexCoord;
layout (location=2) in vec3 vertexNormal;

uniform mat4 motion;
uniform mat4 view;
uniform mat4 projection;

out vec2 fragmentTexCoord;
out vec3 fragmentPosition;
out vec3 fragmentNormal;

void main()
{
    gl_Position = projection * view * motion * vec4(vertexPos, 1.0);
    fragmentTexCoord = vertexTexCoord;
    fragmentPosition = (motion * vec4(vertexPos, 1.0)).xyz;
    fragmentNormal = mat3(motion) * vertexNormal;
}