# ruff: noqa: F405

import glfw
import numpy as np
from OpenGL.GL import *  # noqa: F403
from OpenGL.GL.shaders import compileShader, compileProgram
from PIL import Image
import sys
import pyrr  # For matrix operations
import os
import json
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.options import load_yaml_options
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="OpenGL Tessellation Rendering")
    parser.add_argument(
        "-o", "--option", type=str, required=True, help="Path to options YAML file"
    )
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-r", "--rootpath", type=str)
    parser.add_argument("-g", "--group", type=str)
    parser.add_argument("-s", "--scene", type=str)
    parser.add_argument("-e", "--exp", type=str)
    return load_yaml_options(parser.parse_args())


# A simple OBJ loader that assumes positions, normals and texture coordinates
def load_obj(filename):
    vertices = []
    texcoords = []
    faces = []

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()[1:]
                vertices.append(np.array(list(map(float, parts)), dtype=np.float32))
            elif line.startswith("vt "):
                parts = line.strip().split()[1:]
                texcoords.append([float(part) for part in parts])
            elif line.startswith("f "):
                # Expecting faces like: f v/t v/t v/t or f v// v// v// (without normals)
                face = []
                parts = line.strip().split()[1:]
                for part in parts:
                    vals = part.split("/")
                    # OBJ indices are 1-based
                    vi = int(vals[0]) - 1
                    # Optionally load texcoords if available
                    ti = int(vals[1]) - 1 if len(vals) > 1 and vals[1] != "" else None
                    face.append((vi, ti))
                faces.append(face)

    # Compute normals per vertex if not provided
    vertex_normals = [np.zeros(3, dtype=np.float32) for _ in range(len(vertices))]
    for face in faces:
        # Assuming each face is a triangle
        i0, i1, i2 = face[0][0], face[1][0], face[2][0]
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        # Normalize the face normal to avoid scale issues (if non-zero)
        norm = np.linalg.norm(face_normal)
        if norm != 0:
            face_normal /= norm
        # Add the face normal to each vertex normal
        vertex_normals[i0] += face_normal
        vertex_normals[i1] += face_normal
        vertex_normals[i2] += face_normal

    # Normalize the accumulated vertex normals
    for i in range(len(vertex_normals)):
        norm = np.linalg.norm(vertex_normals[i])
        if norm != 0:
            vertex_normals[i] /= norm
        else:
            # Fallback normal if calculation fails
            vertex_normals[i] = np.array([0, 0, 1], dtype=np.float32)

    # Create interleaved array (position, normal, texcoord)
    data = []
    for face in faces:
        for vi, ti in face:
            pos = vertices[vi].tolist()
            norm = vertex_normals[vi].tolist()
            uv = (
                texcoords[ti]
                if (ti is not None and ti < len(texcoords))
                else [0.0, 0.0]
            )
            data.extend(pos + norm + uv)
    return np.array(data, dtype=np.float32), len(faces) * 3


# Shader sources
vertex_shader_source = """
#version 400 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

out VS_OUT {
    vec3 pos;
    vec3 normal;
    vec2 uv;
} vs_out;

void main()
{
    // Pass attributes to tessellation control stage
    vs_out.pos = in_position;
    vs_out.normal = in_normal;
    vs_out.uv = in_uv;
    gl_Position = vec4(in_position, 1.0);
}
"""

tess_control_shader_source = """
#version 400 core
layout(vertices = 3) out;

in VS_OUT {
    vec3 pos;
    vec3 normal;
    vec2 uv;
} tc_in[];

out TC_OUT {
    vec3 pos;
    vec3 normal;
    vec2 uv;
} tc_out[];

uniform float tessLevel = 4.0; // Adjust tessellation level

void main()
{
    tc_out[gl_InvocationID].pos = tc_in[gl_InvocationID].pos;
    tc_out[gl_InvocationID].normal = tc_in[gl_InvocationID].normal;
    tc_out[gl_InvocationID].uv = tc_in[gl_InvocationID].uv;

    // Set tessellation levels on one invocation (e.g., invocation 0)
    if(gl_InvocationID == 0)
    {
        gl_TessLevelInner[0] = tessLevel;
        gl_TessLevelOuter[0] = tessLevel;
        gl_TessLevelOuter[1] = tessLevel;
        gl_TessLevelOuter[2] = tessLevel;
    }
}
"""

tess_evaluation_shader_source = """
#version 400 core
layout(triangles, equal_spacing, cw) in;

in TC_OUT {
    vec3 pos;
    vec3 normal;
    vec2 uv;
} te_in[];

// out TE_OUT {
//    vec3 normal;
//    vec2 uv;
// } te_out;

out TE_OUT {
    vec3 position_world;
    vec2 uv;
} te_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform sampler2D displacementMap;
uniform float dispScale = 1;



void main()
{
    // Barycentric interpolation of vertex attributes
    vec3 pos = gl_TessCoord.x * te_in[0].pos +
               gl_TessCoord.y * te_in[1].pos +
               gl_TessCoord.z * te_in[2].pos;

    vec3 normal = normalize(gl_TessCoord.x * te_in[0].normal +
                            gl_TessCoord.y * te_in[1].normal +
                            gl_TessCoord.z * te_in[2].normal);

    vec2 uv = gl_TessCoord.x * te_in[0].uv +
             gl_TessCoord.y * te_in[1].uv +
             gl_TessCoord.z * te_in[2].uv;

    // Sample the displacement map
    float displacement = texture(displacementMap, uv).r;
    // Offset the position along the normal direction
    pos += normal * displacement * dispScale;

    gl_Position = projection * view * model * vec4(pos, 1.0);
//    te_out.normal = normal;
//    te_out.uv = uv;

    vec4 pos_world = model * vec4(pos, 1.0);
    gl_Position = projection * view * pos_world;
    te_out.position_world = pos_world.xyz;
    te_out.uv = uv;
}
"""

fragment_shader_source = """
#version 400 core
in TE_OUT {
    vec3 position_world;
    vec2 uv;
} fs_in;

uniform sampler2D normalMap;

uniform vec3 lightDir = normalize(vec3(-0.4, -0.6, 1));
uniform vec3 diffuseColor = vec3(0.8, 0.8, 0.8);
uniform float ambientIntensity = 0.2;
uniform float blendFactor = 0.5;  // 0: pure geometry, 1: pure normal map

out vec4 fragColor;

void main()
{
    // Screen-space differential normal (from geometry)
    vec3 dpdx = dFdx(fs_in.position_world);
    vec3 dpdy = dFdy(fs_in.position_world);
    vec3 normal_geom = normalize(cross(dpdx, dpdy));

    // Normal from normal map (tangent space, assumed aligned with geometry)
    vec3 nmap = texture(normalMap, fs_in.uv).rgb;
    nmap = normalize(nmap * 2.0 - 1.0);

    // Blended normal
    vec3 blendedNormal = normalize(mix(normal_geom, nmap, blendFactor));

    // Lighting
    float diff = max(dot(blendedNormal, lightDir), 0.0);
    float lighting = diff + ambientIntensity;

    fragColor = vec4(diffuseColor * lighting, 1.0);
}
"""


# Compile shader program
def compile_shaders():
    try:
        shader = compileProgram(
            compileShader(vertex_shader_source, GL_VERTEX_SHADER),
            compileShader(tess_control_shader_source, GL_TESS_CONTROL_SHADER),
            compileShader(tess_evaluation_shader_source, GL_TESS_EVALUATION_SHADER),
            compileShader(fragment_shader_source, GL_FRAGMENT_SHADER),
        )
        return shader
    except RuntimeError as e:
        print("Shader compilation error:", e)
        sys.exit(1)


# Load texture from an image file (normal map)
def load_texture_image(path):
    img = Image.open(path).convert("RGB")
    img_data = np.array(img, dtype=np.uint8)
    width, height = img.size

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data
    )
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return texture


# Load texture from a displacement npy file
def load_texture_npy(path):
    disp_map = np.load(path)
    print("Displacement stats:", disp_map.min(), disp_map.max())
    if disp_map.dtype != np.float32:
        disp_map = disp_map.astype(np.float32)

    height, width = disp_map.shape
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Use single channel float texture
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, disp_map
    )
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return texture


def main():
    opt = parse_args()
    opt.output_path = os.path.join(opt.group, opt.exp)

    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    # Request an OpenGL 4.0 context and make window invisible
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    width, height = 1080, 1080
    window = glfw.create_window(
        width, height, "Headless Tessellation Rendering", None, None
    )
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        sys.exit(1)
    glfw.make_context_current(window)

    # Compile shader program
    shader = compile_shaders()

    # Load mesh and create VAO/VBO
    mesh_path = os.path.join(opt.output_path, "mesh.obj")
    mesh_data, vertex_count = load_obj(mesh_path)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, mesh_data.nbytes, mesh_data, GL_STATIC_DRAW)

    # The layout is: position (3 floats), normal (3 floats), uv (2 floats)
    stride = (3 + 3 + 2) * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Load textures
    normal_texture = load_texture_image(os.path.join(opt.output_path, "normal.png"))
    displacement_texture = load_texture_npy(
        os.path.join(opt.output_path, "displacement.npy")
    )

    glEnable(GL_DEPTH_TEST)
    model = pyrr.matrix44.create_identity(dtype=np.float32)
    if opt.dataset == "blender":
        with open(f"{opt.rootpath}/{opt.scene}/transforms_train.json", "r") as f:
            transforms = json.load(f)
        poses = np.array([x["transform_matrix"] for x in transforms["frames"]])
        fov_x = transforms["camera_angle_x"]
        fov_y = (2 * math.atan(math.tan(fov_x / 2) * (height / width))) * 180 / math.pi
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=fov_y, aspect=width / height, near=0.1, far=100, dtype=np.float32
        )
        pose = poses[opt.vis_idx[0]]
        eye = pose[:3, 3]
        look_at = pose[:3, 3] - pose[:3, 2]
        up = pose[:3, 1]
        view = pyrr.matrix44.create_look_at(
            eye=eye, target=look_at, up=up, dtype=np.float32
        )
    elif opt.dataset == "avatarhq":
        view = pyrr.matrix44.create_look_at(
            eye=[2, 1.5, 4], target=[0, 0.9, 0], up=[0, 1, 0], dtype=np.float32
        )
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=25, aspect=width / height, near=0.1, far=100, dtype=np.float32
        )
    # Render one frame
    glViewport(0, 0, width, height)
    glClearColor(1, 1, 1, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader)
    # Set uniforms for transformations
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
    glUniformMatrix4fv(
        glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection
    )

    # Bind textures
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, displacement_texture)
    glUniform1i(glGetUniformLocation(shader, "displacementMap"), 0)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, normal_texture)
    glUniform1i(glGetUniformLocation(shader, "normalMap"), 1)

    # Bind VAO and draw using patches (each patch is a triangle)
    glBindVertexArray(vao)
    glPatchParameteri(GL_PATCH_VERTICES, 3)
    glDrawArrays(GL_PATCHES, 0, vertex_count)
    glBindVertexArray(0)
    glFinish()

    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(f"{opt.output_path}/geom-output.png")
    print("Rendered image saved as", f"{opt.output_path}/geom-output.png")

    glfw.terminate()


if __name__ == "__main__":
    main()
