#version 330 core

in vec2 fragmentTexCoord;
in vec3 fragmentPosition;
in vec3 fragmentNormal;

struct LightSource {
    vec3 position;
    vec3 color;
    float strength;
    float ambientStrength;
    float minDistance;
    float maxDistance;
};

struct Material {
    vec3 ambientWeighting;
    vec3 diffuseWeighting;
    vec3 specularWeighting;
    float specularExponent;
    float opacity;
    float specularTint;
};

uniform sampler2D imageTexture;
uniform LightSource lightSource;
uniform Material currentMaterial;
uniform vec3 cameraPosition;

out vec4 color;

void main()
{
    color = vec4(abs(fragmentNormal), 1);
}
