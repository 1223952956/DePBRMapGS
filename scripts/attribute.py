import os
import sys
import numpy as np
from PIL import Image
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.options import load_yaml_options
import argparse

import torch

torch.set_default_device("cuda:0")


def parse_args():
    parser = argparse.ArgumentParser(description="OpenGL Tessellation Rendering")
    parser.add_argument(
        "-o", "--option", type=str, required=True, help="Path to options YAML file"
    )
    parser.add_argument("-g", "--group", type=str)
    parser.add_argument("-e", "--exp", type=str)
    return load_yaml_options(parser.parse_args())


def calculate_normal(verts):
    edge1 = verts[:, 1] - verts[:, 0]
    edge2 = verts[:, 2] - verts[:, 0]
    cross = torch.cross(edge1, edge2, dim=1)
    normals = torch.nn.functional.normalize(cross, dim=1)
    return normals


def rotate_back_normal_map(normal_map):
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

    normal_map = np.einsum("ij,hwj->hwi", R, normal_map)  # apply rotation
    return normal_map


if __name__ == "__main__":
    opt = parse_args()
    file_path = os.path.join(opt.group, opt.exp)
    output_path = os.path.join(file_path, "blender")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # === 1. Flip texture map vertically ===
    texture_img = Image.open(f"{file_path}/texture2.png")
    texture_img = texture_img.transpose(Image.FLIP_TOP_BOTTOM)
    texture_img.save(f"{output_path}/texture_blender.png")

    # === 2. Convert and flip normal map ===

    normal_img = imageio.v2.imread(f"{file_path}/normal.png").astype(np.float32) / 255.0
    blender_normal = np.zeros_like(normal_img[..., :3])
    blender_normal[..., 0] = normal_img[..., 0]  # X stays
    blender_normal[..., 1] = 1.0 - normal_img[..., 2]  # Y = -Z
    blender_normal[..., 2] = normal_img[..., 1]  # Z = Y
    blender_normal = blender_normal[::-1]
    blender_normal_uint8 = (blender_normal * 255).astype(np.uint8)
    Image.fromarray(blender_normal_uint8).save(f"{output_path}/normal_blender.png")

    # === 3. Convert and flip displacement map ===
    displacement = np.load(f"{file_path}/displacement.npy")
    disp_min, disp_max = displacement.min(), displacement.max()
    disp_max_max = max(abs(disp_min), abs(disp_max))
    displacement_norm = (displacement + disp_max_max) / (2 * disp_max_max)
    print(disp_max_max)
    displacement_norm = displacement_norm[::-1]  # vertical flip
    displacement_uint16 = (displacement_norm * 65535).astype(np.uint16)
    imageio.imwrite(f"{output_path}/displacement_blender.png", displacement_uint16)

    print("All maps converted successfully.")
