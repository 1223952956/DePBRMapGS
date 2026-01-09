import numpy as np
import torch
import random
import argparse
from options.options import load_yaml_options
from dataloader import load_dataset
from gaussians.render_2dgs import render3
from os.path import join
from utils.common import load_mesh, pixels_barycentric_coordinates
from PIL import Image
from gaussians.gaussian_model import GaussianModelVis
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


def add_arguments(parser):
    parser.add_argument("-e", "--exp", type=str)
    parser.add_argument("-o", "--option", type=str)
    parser.add_argument("-r", "--rootpath", type=str)


def calculate_normal(verts):
    edge1 = verts[:, 1] - verts[:, 0]
    edge2 = verts[:, 2] - verts[:, 0]
    cross = torch.cross(edge1, edge2, dim=1)
    normals = torch.nn.functional.normalize(cross, dim=1)
    return normals


def get_points(opt, mesh_path, displace_path):
    displacement = np.load(displace_path)
    displacement = torch.from_numpy(displacement).cuda()
    size_ = displacement.shape[0]
    vertices, faces, faces_uvs, uvs = load_mesh(mesh_path)
    pixels, barycentric_coordinates, indices, uvs_img = pixels_barycentric_coordinates(
        uvs[faces_uvs], size_
    )
    vertces_pose = vertices[faces]
    pixel_vertices = vertces_pose[indices]
    pixel_3d = (pixel_vertices * barycentric_coordinates[..., None]).sum(1)
    normals = calculate_normal(pixel_vertices)
    mesh = edict(
        indices=indices,
        face=faces,
        barycentrics=barycentric_coordinates,
        pixels=pixels,
        uvs=uvs_img,
    )
    return mesh, pixels, pixel_3d, displacement, normals


def load_result_mesh(opt):
    model_path = join(opt.group, opt.exp)
    displace_path = join(model_path, "displacement.npy")
    mesh_path = join(model_path, "mesh.obj")
    mesh, pixels, pixel_3d, displ, normals = get_points(opt, mesh_path, displace_path)
    texture_path = join(model_path, "texture.png")
    texture = Image.open(texture_path)
    original_w, original_h = texture.size
    texture = np.array(texture) / 255.0
    texture = torch.from_numpy(texture).cuda()
    return mesh, texture, pixel_3d, pixels, displ, normals


def torch_grid_sample(indices, image, w, h):
    grid = indices.float().clone()
    grid[:, 0] = (grid[:, 0] / (w - 1)) * 2 - 1
    grid[:, 1] = (grid[:, 1] / (h - 1)) * 2 - 1
    grid = grid.unsqueeze(0).unsqueeze(0)
    image_reshaped = image.unsqueeze(0)
    sampled = torch.nn.functional.grid_sample(image_reshaped, grid, align_corners=True)
    return sampled.squeeze()


def reproject_texture(texture, pixel_3d, pixels, camera, depth_map, scale=1):
    pixel_3d_h = torch.cat(
        [pixel_3d, torch.ones(pixel_3d.shape[0], 1, device=pixel_3d.device)], dim=1
    )
    pixel_camera_space = (camera.extr @ pixel_3d_h.T).T[:, :3]

    visible_mask = pixel_camera_space[:, 2] > 0
    pixel_camera_space = pixel_camera_space[visible_mask]
    visible_pixels = pixels[visible_mask]

    pixel_image_space = (camera.intr @ pixel_camera_space.T).T
    pixel_image_space = pixel_image_space[:, :2] / pixel_image_space[:, 2:3]
    pixel_image_space = pixel_image_space

    h, w = camera["height"], camera["width"]
    valid_mask = (
        (pixel_image_space[:, 0] >= 0)
        & (pixel_image_space[:, 0] < w)
        & (pixel_image_space[:, 1] >= 0)
        & (pixel_image_space[:, 1] < h)
    )

    pixel_image_space = pixel_image_space[valid_mask]
    visible_pixels = visible_pixels[valid_mask]
    pixel_camera_space = pixel_camera_space[valid_mask]

    sampled_depth = torch_grid_sample(pixel_image_space, depth_map, w, h)
    depth_diff = (sampled_depth - pixel_camera_space[:, 2]).abs()
    filtered = depth_diff < 0.05 * scale
    visible_pixels = visible_pixels[filtered]
    pixel_image_space = pixel_image_space[filtered]
    pixel_camera_space = pixel_camera_space[filtered]

    colors = texture[visible_pixels[:, 1], visible_pixels[:, 0]]
    return colors, pixel_image_space, visible_pixels, depth_diff[filtered]


@torch.no_grad()
def render(gaussian_vals, calib, bg):
    render_ret = render3(
        gaussian_vals, bg, calib["extr"], calib["intr"], calib["width"], calib["height"]
    )
    return render_ret


def train(opt, train_data, gaussian_model, texture, displ, pixel_3d, pixels, normals):
    opt.threed = False
    gaussian_model.threed = False
    texture = texture.float()
    texture_opt = torch.nn.Parameter(texture, requires_grad=True)
    displ = torch.nn.Parameter(displ.float(), requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": texture_opt, "lr": 0.001},
        ]
    )
    progress = tqdm(range(1000), file=sys.stderr)
    stack = []
    for iter in progress:
        optimizer.zero_grad()
        if stack == []:
            stack = torch.randperm(len(train_data)).squeeze().tolist()

        idx = stack.pop()
        data = train_data[idx]
        bg = torch.ones(3).float().cuda()
        gaussian_vals = gaussian_model.get_gaussian_vals()
        render_ret = render(gaussian_vals, data.camera, bg)
        depth = render_ret["surf_depth"]
        curr_pixel = pixel_3d + normals * displ[pixels[:, 1], pixels[:, 0]][:, None]
        colors, pixel_image_space, visible_pixels, depth_diff = reproject_texture(
            texture_opt, curr_pixel, pixels, data.camera, depth, opt.pos_scale
        )
        gt_image = data.img
        alpha_chanel = data.mask.float()
        gt_image = torch.cat([gt_image, alpha_chanel[..., None]], dim=2)
        gt_colors = torch_grid_sample(
            pixel_image_space,
            gt_image.permute(2, 0, 1),
            data.camera["width"],
            data.camera["height"],
        ).T
        visible = gt_colors[..., 3] > 0.5
        # colors[visible] = gt_colors[visible] # directly use the color
        loss = (
            torch.nn.functional.mse_loss(colors[visible], gt_colors[visible])
            + depth_diff.mean()
        )
        loss.backward()
        optimizer.step()
    plt.imsave(
        f"{opt.group}/{opt.exp}/texture2.png",
        texture_opt.clamp(0, 1).detach().cpu().numpy(),
    )
    np.save(f"{opt.group}/{opt.exp}/displacement2.npy", displ.detach().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    opt_arg = parser.parse_args()
    opt = load_yaml_options(opt_arg)
    opt.sh = False
    random.seed(27519)
    np.random.seed(27519)
    torch.manual_seed(27519)
    torch.set_default_device(f"cuda:{opt.cuda}")

    train_data = load_dataset(opt)
    mesh, texture, pixel_3d, pixels, displ, normals = load_result_mesh(opt)

    gaussian_model = GaussianModelVis(
        opt, f"{opt.group}/{opt.exp}/model.ckpt", mesh, False
    )
    train(opt, train_data, gaussian_model, texture, displ, pixel_3d, pixels, normals)
