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

vec4 calculatePointLight(vec3 fragmentPosition, vec3 fragmentNormal);

void main()
{
    color = calculatePointLight(fragmentPosition, fragmentNormal);
}

vec4 calculatePointLight(vec3 fragmentPosition, vec3 fragmentNormal) {
    vec3 result = vec3(0.0);
    vec4 baseTexture = texture(imageTexture, fragmentTexCoord).rgba;
    vec3 textureColor = baseTexture.rgb;

    // Geometry Data
    vec3 fragLight = lightSource.position - fragmentPosition;
    float dist2inv = length(fragLight);
    // Get a distance modifier from the light source (using player distance from light)
    float distModifier = 1 - min(max(dist2inv - lightSource.minDistance, 0.0) / lightSource.maxDistance, 1.0);

    dist2inv = 1 / pow(dist2inv, 2);
    fragLight = normalize(fragLight);

    // Diffuse lighting
    result += textureColor * lightSource.color * currentMaterial.diffuseWeighting * lightSource.strength * distModifier * max(0.0, dot(fragmentNormal, fragLight)) * dist2inv;

    return vec4(result, baseTexture.a);
}
