ARENA_FRAG_SHADER = '''
#version 330

uniform vec3 ballPos;

in FD {
    vec3 pos;
    vec3 norm;
    vec2 text;
} inData;

out vec4 f_color;

bool IsPosInGoal(vec3 pos) {
    float epsilon = 5.0;
    return abs(pos.y) > (5120.0 + epsilon) && abs(pos.y) < 6500 && pos.z < 1000;
}

vec3 TeamColorFromPos(vec3 pos) {
    vec3 blue = vec3(0.0, 0.3, 1.0);
    vec3 orange = vec3(1.0, 0.3, 0.0);
    vec3 gray = vec3(0.5);
    float blueScale = clamp(-pos.y / 4000.0, 0, 1);
    float orangeScale = clamp(pos.y / 4000.0, 0, 1);

    if (!IsPosInGoal(pos)) {
        blueScale *= 0.7;
        orangeScale *= 0.7;
    }

    float grayScale = 1.0 - (blueScale + orangeScale);
    return blue * blueScale + orange * orangeScale + gray * grayScale;
}

float GetGridLineVal(float gridSize, float gridLineWidth, vec3 pos, vec3 norm, vec3 pixelWidths) {
    vec3 gridPixelWidth = pixelWidths / gridSize;
    vec3 gridPos = pos / gridSize;
    vec3 distToLine = abs(mod(gridPos + 0.5, 1.0) - 0.5);
    float halfWidth = (gridLineWidth / 2.0) / gridSize;
    vec3 lineFrac = smoothstep(halfWidth + gridPixelWidth, halfWidth - gridPixelWidth, distToLine);

    lineFrac *= clamp(1.0 - abs(norm), 0.0, 1.0);
    return max(lineFrac.x, max(lineFrac.y, lineFrac.z));
}

void main() {
    vec3 pixelWidths = fwidth(inData.pos);
    float verticalFrac = abs(inData.norm.z);
    float gridSize = 1600;
    float gridLineWidth = 10;

    if (IsPosInGoal(inData.pos)) {
        gridSize = 100;
    } else if (inData.pos.z < 5) {
        gridSize /= 2;
    }

    float gridVal = GetGridLineVal(gridSize, gridLineWidth, inData.pos, inData.norm, pixelWidths);
    gridVal += GetGridLineVal(gridSize / 2, gridLineWidth, inData.pos, inData.norm, pixelWidths) * 0.25;
    gridVal += GetGridLineVal(gridSize / 4, gridLineWidth / 2, inData.pos, inData.norm, pixelWidths) * 0.125;
    gridVal = min(gridVal, 1);

    float minGridLight = 0.08 + abs(inData.norm.z) * 0.12;
    float gridLight = gridVal * (0.8 - minGridLight) + minGridLight;
    float lightFactor = (dot(inData.norm, normalize(vec3(0.3, -0.2, 0.6))) + 1) / 2;
    float brightness = lightFactor * 0.7 + 0.3;
    float verticalGridColorScale = 0.3;
    float teamColorFrac = gridVal * (((1.0 - verticalFrac) * verticalGridColorScale) + (1.0 - verticalGridColorScale));
    vec3 teamColor = TeamColorFromPos(inData.pos * teamColorFrac);

    f_color = vec4(gridLight * teamColor * brightness, 1.0);

    if (inData.norm.z > 0.5 && inData.pos.z < ballPos.z - 100) {
        float heightRatio = clamp((ballPos.z - 300) / (900 - 300), 0, 1);
        float dist = length(inData.pos.xy - ballPos.xy);
        if (dist < 100 && dist > 90)
            f_color.rgb += vec3(0.5 * heightRatio);
        if (dist < 200 && dist > 190)
            f_color.rgb += vec3(0.25 * heightRatio);
    }
}
'''

ARENA_VERT_SHADER = '''
#version 330

uniform mat4 m_vp;
uniform mat4 m_model;

in vec3 in_position;
in vec4 in_normal;
in vec2 in_texcoord_0;

out FD {
    vec3 pos;
    vec3 norm;
    vec2 text;
} outData;

void main() {
    outData.pos = in_position;
    outData.norm = normalize((m_model * vec4(in_normal.xyz, 0.0)).xyz);
    outData.text = in_texcoord_0;
    gl_Position = m_vp * m_model * vec4(in_position, 1.0);
}
'''
