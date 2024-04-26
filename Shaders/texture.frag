#version 330 core

in vec2 fragmentTextCoord;

out vec4 color;

uniform sampler2D imageTexture;

void main() {
    vec4 texColor = texture(imageTexture, fragmentTextCoord);
    color = vec4(texColor.b, texColor.g, texColor.r, texColor.a);
}
