# ruff: noqa: F405

import glfw
import numpy as np
from OpenGL.GL import *  # noqa: F403
from OpenGL.GL.shaders import compileShader, compileProgram
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyrr
import os
import math
import json
import argparse
from options.options import load_yaml_options


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenGL Tessellation Rendering with Textures"
    )
    parser.add_argument(
        "-o", "--option", type=str, required=True, help="Path to options YAML file"
    )
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-r", "--rootpath", type=str)
    parser.add_argument("-g", "--group", type=str)
    parser.add_argument("-s", "--scene", type=str)
    parser.add_argument("-e", "--exp", type=str)
    parser.add_argument(
        "--gt", action="store_true", help="Use ground truth camera pose"
    )
    parser.add_argument(
        "--static", action="store_true", help="whether use static tessellation"
    )
    return load_yaml_options(parser.parse_args())


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
        norm = np.linalg.norm(face_normal)
        if norm != 0:
            face_normal /= norm
        vertex_normals[i0] += face_normal
        vertex_normals[i1] += face_normal
        vertex_normals[i2] += face_normal

    # Normalize the accumulated vertex normals
    for i in range(len(vertex_normals)):
        norm = np.linalg.norm(vertex_normals[i])
        if norm != 0:
            vertex_normals[i] /= norm
        else:
            vertex_normals[i] = np.array([0, 0, 1], dtype=np.float32)

    # Create interleaved array (position, normal, texcoord)
    data = []
    # Normalize vertices (x, y) axis to be centered to (0, 0)
    vertices = np.array(vertices)
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


vertex_shader_source_s = """
#version 420 core
layout(location = 0) in vec3 in_position;
// glClearColor(0, 0, 0, 1)
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec2 vUV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vNormal    = in_normal;
    vUV        = in_uv;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
"""

fragment_shader_source_s = """
#version 420 core
layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec2 vUV;

uniform sampler2D textureMap;
uniform vec3        lightDir         = normalize(vec3(0.5, -0.5, 0));
uniform float       ambientIntensity = 1;

out vec4 fragColor;

void main() {
    vec3 texColor = texture(textureMap, vUV).rgb;
    vec3 N        = normalize(vNormal);
    // float diff    = max(dot(N, lightDir), 0.0);
    float lit     = ambientIntensity;
    fragColor     = vec4(texColor * lit, 1.0);
}
"""

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

out TE_OUT {
    vec3 normal;
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
    te_out.normal = normal;
    te_out.uv = uv;
}
"""

fragment_shader_source = """
#version 400 core
in TE_OUT {
    vec3 normal;
    vec2 uv;
} fs_in;

uniform sampler2D normalMap;
uniform sampler2D textureMap;
uniform vec3 lightDir = normalize(vec3(0.5, -0.5, 0));
uniform float ambientIntensity = 1; // Ambient light added to dark areas

out vec4 fragColor;

void main()
{
    // Sample the normal map and transform it from [0,1] to [-1,1]
    vec3 nmap = texture(normalMap, fs_in.uv).rgb;
    nmap = normalize(nmap * 2.0 - 1.0);
    
    // Combine the geometry normal and the normal map (simple blend)
    vec3 finalNormal = normalize(fs_in.normal + nmap);
    float diff = max(dot(finalNormal, lightDir), 0.0);
    
    // Add ambient light so that dark areas are brighter
    float lighting =  ambientIntensity;
    
    // Sample the texture map (RGBA)
    vec4 textureColor = texture(textureMap, fs_in.uv);
    // if (textureColor.a < 0.5) discard;
    // Multiply texture color with combined lighting and preserve alpha
    fragColor = vec4(textureColor.rgb * lighting, textureColor.a);
}
"""


# Compile shader program
def compile_shaders(static=False):
    try:
        if static:
            shader = compileProgram(
                compileShader(vertex_shader_source_s, GL_VERTEX_SHADER),
                compileShader(fragment_shader_source_s, GL_FRAGMENT_SHADER),
            )
        else:
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


# Load texture from an RGB image file (for normal maps)
def load_texture_image(path):
    img = Image.open(path).convert("RGB")
    img_data = np.array(img, dtype=np.uint8)[1:, 1:]
    width, height = img_data.shape[:2]
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


# Load texture from a 4-channel RGBA image file (for diffuse texture maps)
def load_texture_rgba_image(path):
    img = Image.open(path).convert("RGBA")
    img_data = np.array(img, dtype=np.uint8)[2:-1, 2:-1]
    width, height = img_data.shape[:2]
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data
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
    if opt.static:
        print("using static tessllation")
    else:
        print("using dynamic tessellation")
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    # Request an OpenGL 4.0 context and make the window invisible
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
    shader = compile_shaders(opt.static)

    # Load mesh and create VAO/VBO
    if not opt.static:
        mesh_path = os.path.join(opt.output_path, "mesh.obj")
    else:
        mesh_path = os.path.join(opt.output_path, "mesh_displaced3.obj")
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
    if not opt.static:
        displacement_texture = load_texture_npy(
            os.path.join(opt.output_path, "displacement.npy")
        )
    diffuse_texture = load_texture_rgba_image(
        os.path.join(opt.output_path, "texture2.png")
    )

    glEnable(GL_DEPTH_TEST)

    if opt.dataset == "blender":
        if opt.gt:
            with open(f"{opt.rootpath}/{opt.scene}/transforms_train.json", "r") as f:
                transforms = json.load(f)
            poses = np.array([x["transform_matrix"] for x in transforms["frames"]])
            fov_x = transforms["camera_angle_x"]
            fov_y = (
                (2 * math.atan(math.tan(fov_x / 2) * (height / width))) * 180 / math.pi
            )
            frame_range = range(80, 100)
        else:
            fov_y = 30
            frame_range = range(120)
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=fov_y, aspect=width / height, near=0.1, far=100, dtype=np.float32
        )

    elif opt.dataset == "avatarhq":
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=25, aspect=width / height, near=0.1, far=100, dtype=np.float32
        )
        frame_range = range(120)

    output_dir = os.path.join(opt.output_path, "tex_frames")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for n, i in enumerate(frame_range):
        if opt.dataset == "blender":
            if opt.gt:
                pose = poses[i]
                eye = pose[:3, 3]
                look_at = pose[:3, 3] - pose[:3, 2]
                up = pose[:3, 1]
            else:
                angle = 2 * math.pi * i / len(frame_range)
                radius = math.sqrt(10)
                eye_x = radius * math.cos(angle)
                eye_y = radius * math.sin(angle)
                eye_z = 2.8
                eye = [eye_x, eye_y, eye_z]
                look_at = [0, 0, 0]
                up = [0, 0, 1]
        elif opt.dataset == "avatarhq":
            angle = 2 * math.pi * i / len(frame_range)
            radius = math.sqrt(20)
            eye_x = radius * math.cos(angle)
            eye_y = 1.5
            eye_z = radius * math.sin(angle)
            eye = [eye_x, eye_y, eye_z]
            look_at = [0, 0.9, 0]
            up = [0, 1, 0]

        view = pyrr.matrix44.create_look_at(
            eye=eye, target=look_at, up=up, dtype=np.float32
        )
        glViewport(0, 0, width, height)
        glClearColor(0, 0, 0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)
        model = pyrr.matrix44.create_identity(dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(
            glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection
        )

        if not opt.static:
            # Bind displacement texture to texture unit 0
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, displacement_texture)
            glUniform1i(glGetUniformLocation(shader, "displacementMap"), 0)
            # glEnable(GL_BLEND)
            # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            # Bind diffuse texture (RGBA) to texture unit 2
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, diffuse_texture)
            glUniform1i(glGetUniformLocation(shader, "textureMap"), 2)
        else:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, diffuse_texture)
            glUniform1i(glGetUniformLocation(shader, "textureMap"), 0)

        glBindVertexArray(vao)
        if not opt.static:
            glPatchParameteri(GL_PATCH_VERTICES, 3)
            glDrawArrays(GL_PATCHES, 0, vertex_count)
        else:
            glDrawArrays(GL_TRIANGLES, 0, vertex_count)
        glBindVertexArray(0)
        # Ensure all commands are finished
        glFinish()
        # Read the pixels from the framebuffer
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (width, height), pixels)
        # Flip vertically to correct the orientation (OpenGL's origin is bottom-left)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        frame_filename = os.path.join(output_dir, f"frame_{i:03d}.png")
        image.save(frame_filename)
        print(f"Saved {frame_filename}")

    glfw.terminate()

    if not opt.gt:
        # Combine frames into a video using ffmpeg (ensure ffmpeg is installed)
        ffmpeg_cmd = (
            f"ffmpeg -y -framerate 30 -i {opt.output_path}/tex_frames/frame_%03d.png "
            f"-c:v libx264 -pix_fmt yuv420p {opt.output_path}/tex_video.mp4"
        )
        os.system(ffmpeg_cmd)
        print(f"Video saved as {opt.output_path}/tex_video.mp4")


if __name__ == "__main__":
    main()
