BOOST_VERT_SHADER = '''
#version 330

in vec2 in_position;
in vec2 in_texcoord;
out vec2 texcoord;

void main() {
    texcoord = in_texcoord;
    gl_Position = vec4(in_position, 0, 1);
}
'''

BOOST_FRAG_SHADER = '''
#version 330

uniform sampler2D Texture;
in vec2 texcoord;
out vec4 f_color;

void main() {
    f_color = texture(Texture, texcoord);
}
'''

FRAG_SHADER = '''
#version 330

uniform sampler2D Texture;
uniform vec4 globalColor;

in FD {
    vec3 norm;
    vec2 text;
} inData;

out vec4 f_color;

void main() {
    if (globalColor.a > 0) {
        f_color = globalColor;
        return;
    }

    vec4 textureColor = texture(Texture, inData.text);
    if (textureColor.a < 1) {
        f_color = textureColor;
        return;
    }

    vec3 ambientColor = vec3(0.7, 0.85, 1.0);
    vec3 sunColor = vec3(1.0, 0.85, 0.7);
    vec3 lightDir = normalize(vec3(0.1, 0.2, 1.0));
    float lightAlign = max(dot(lightDir, normalize(inData.norm)), 0);
    vec3 lightColor = ambientColor * 0.4 + sunColor * lightAlign;

    f_color = textureColor * vec4(lightColor, 1.0);
}
'''

VERT_SHADER = '''
#version 330

uniform mat4 m_vp;
uniform mat4 m_model;

in vec3 in_position;
in vec4 in_normal;
in vec2 in_texcoord_0;

out FD {
    vec3 norm;
    vec2 text;
} outData;

void main() {
    outData.norm = (m_model * vec4(in_normal.xyz, 0.0)).xyz;
    outData.text = in_texcoord_0;
    gl_Position = m_vp * m_model * vec4(in_position, 1.0);
}
'''
