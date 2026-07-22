#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics import focal2fov, getProjectionMatrix
from utils.sh import eval_sh
from utils.pbr import shade_pbr


def render3(
    gaussian_vals: dict,
    bg_color: torch.Tensor,
    extr: torch.Tensor,
    intr: torch.Tensor,
    img_w: int,
    img_h: int,
    scaling_modifier=1.0,
    lighting=None
):
    means3D = gaussian_vals["positions"]
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:  # noqa: E722
        pass
    means2D = screenspace_points
    opacity = gaussian_vals["opacity"]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    scales = gaussian_vals["scales"]
    rotations = gaussian_vals["rotations"]

    # Set up rasterization configuration
    FoVx = focal2fov(intr[0, 0].item(), img_w)
    FoVy = focal2fov(intr[1, 1].item(), img_h)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    world_view_transform = extr.transpose(1, 0).cuda()
    projection_matrix = (
        getProjectionMatrix(
            znear=0.1, zfar=100, fovX=FoVx, fovY=FoVy, K=intr, img_w=img_w, img_h=img_h
        )
        .transpose(0, 1)
        .cuda()
    )
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = torch.linalg.inv(extr)[:3, 3]

    raster_settings = GaussianRasterizationSettings(
        image_height=img_h,
        image_width=img_w,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=gaussian_vals["max_sh_degree"],
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    assert not ("colors" in gaussian_vals and "shs" in gaussian_vals), (
        "Cannot use both color and SH!"
    )
    if "albedo" in gaussian_vals:
        colors_precomp = shade_pbr(
            positions=means3D,
            normals=gaussian_vals["normals"],
            albedo=gaussian_vals["albedo"],
            roughness=gaussian_vals["roughness"],
            metallic=gaussian_vals["metallic"],
            camera_position=camera_center,
            lighting=lighting,
        )
    elif "colors" in gaussian_vals:
        colors_precomp = gaussian_vals["colors"]
    else:
        colors_precomp = None
    if "shs" in gaussian_vals:
        shs_view = gaussian_vals["shs"].transpose(1, 2).view(-1, 3, 16)
        dir_pp = means3D - camera_center.repeat(means3D.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussian_vals["max_sh_degree"], shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    shs = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    # additional regularizations
    render_alpha = allmap[1:2]
    render_normal = allmap[2:5]
    # render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = render_depth_expected / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    # get depth distortion map
    render_dist = allmap[6:7]
    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    depth_ratio = 0.5
    surf_depth = (
        render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median
    )

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    viewpoint_camera = Camera(world_view_transform, full_proj_transform, img_w, img_h)
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update(
        {
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
        }
    )
    return rets


def depth_to_normal(view, depth):
    """
    view: view camera
    depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = (
        torch.tensor([[W / 2, 0, 0, (W) / 2], [0, H / 2, 0, (H) / 2], [0, 0, 0, 1]])
        .float()
        .cuda()
        .T
    )
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(),
        torch.arange(H, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
        -1, 3
    )
    rays_d = points @ intrins.inverse().T
    points = depthmap.reshape(-1, 1) * rays_d
    return points


class Camera:
    def __init__(
        self, world_view_transform, full_proj_transform, image_width, image_height
    ):
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.image_width = image_width
        self.image_height = image_height
