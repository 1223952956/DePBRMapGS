import torch
import numpy as np
import pytorch3d
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    RasterizationSettings,
    PerspectiveCameras,
    TexturesUV,
    OrthographicCameras,
    DirectionalLights,
    Materials,
    blending,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex
from PIL import ImageColor
import pymeshlab

ms = pymeshlab.MeshSet()


def create_mesh_with_texture(vertices, faces, uv_verts, uv_faces, texture_map):
    textures = TexturesUV(maps=texture_map, faces_uvs=uv_faces, verts_uvs=uv_verts)
    mesh = Meshes(verts=vertices, faces=faces, textures=textures)
    return mesh


def white_face_mesh(vertices, faces):
    device = vertices.device
    verts_rgb = torch.ones_like(vertices).to(device)
    textures = TexturesVertex(verts_features=verts_rgb)
    return Meshes(verts=vertices, faces=faces, textures=textures).to(device)


def detect_intersections(vertices, faces):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=vertices[0].detach().cpu().numpy(),
            face_matrix=faces[0].detach().cpu().numpy(),
        )
    )
    ms.apply_filter("compute_selection_by_self_intersections_per_face")
    face_mask = torch.from_numpy(ms.current_mesh().face_selection_array()).cuda()
    return torch.nonzero(face_mask).flatten()


def create_perspective_camera(camera_p, device, reso=4):
    intr, width, height = camera_p.intr, camera_p.width, camera_p.height
    fx, fy, px, py = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    translation, rotation = camera_p.extr[:3, 3][None], camera_p.extr[:3, :3][None]
    cameras = PerspectiveCameras(
        focal_length=((-fx * reso, -fy * reso),),
        device=device,
        principal_point=((px * reso, py * reso),),
        R=torch.inverse(rotation),
        T=translation,
        in_ndc=False,
    )
    return cameras, height * reso, width * reso


def render_mesh(
    vertices,
    faces,
    translation,
    rotation,
    color,
    intrinsics,
    device=None,
    shader=1,
    size=1024,
):
    if device is None:
        device = vertices.device

    verts_rgb = color.to(device)  # (B, V, 3)

    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(verts=vertices.to(device), faces=faces.to(device), textures=textures)
    if intrinsics is None:
        width = size
        height = size
        cameras = OrthographicCameras(
            focal_length=((width, height),),
            principal_point=((width / 2.0, height / 2.0),),
            R=rotation,
            T=translation,
            in_ndc=False,
            image_size=((width, height),),
            device=device,
        )
    else:
        intr, width, height = intrinsics
        fx, fy, px, py = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        cameras = PerspectiveCameras(
            focal_length=((-fx, -fy),),
            device=device,
            principal_point=((px, py),),
            R=torch.inverse(rotation),
            T=translation,
            in_ndc=False,
        )

    raster_settings = RasterizationSettings(
        image_size=(height, width),  # (H, W)
        blur_radius=np.log(1.0 / 1e-4) * 1e-7,
        faces_per_pixel=30,
        bin_size=-1,
    )
    if shader == 1:
        blendparam = BlendParams(
            1e-4, 1e-8, np.array(ImageColor.getrgb("black")) / 255.0
        )
        s = Shader(blend_params=blendparam)
    elif shader == 2:
        lights = DirectionalLights(device=device, direction=((0, 0, -1),))
        materials = Materials(
            ambient_color=((1, 1, 1),),
            diffuse_color=((1, 1, 1),),
            specular_color=((1, 1, 1),),
            shininess=64,
            device=device,
        )
        s = SoftPhongShader(
            device=device, cameras=cameras, lights=lights, materials=materials
        )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=s,
    )

    imgs = renderer(mesh, image_size=(height, width))
    return imgs


def softmax_alpha_blend(alpha, fragments, blend_params):
    dists = fragments.zbuf
    dists = torch.where(dists < 0, torch.tensor(float("inf")).to(dists.device), dists)

    z_inv = 1.0 / (
        dists - dists.min(dim=-1, keepdim=True).values + 1e-6
    )  # Stabilize inversion
    z_inv_max = z_inv.max(dim=-1, keepdim=True).values
    z_inv = z_inv - z_inv_max
    weights = torch.nn.functional.softmax(
        z_inv / blend_params.sigma, dim=-1
    )  # Softmax weights
    weights[weights.isnan()] = 0

    blended_alpha = torch.sum(weights * alpha, dim=-1)  # Blended alpha

    return blended_alpha


class Shader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = (
            blend_params
            if blend_params is not None
            else pytorch3d.renderer.BlendParams()
        )

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)

        texels = meshes.sample_textures(fragments)

        background_color = torch.tensor([1, 1, 1, 0]).float().to(texels.device)
        if texels.shape[-1] == 4:
            alpha = texels[..., 3]
            rgb_colors = texels[..., :3]

            rgb_colors = (
                rgb_colors * alpha[..., None]
            )  # (N, H, W, K, 3) * (N, H, W, K, 1)
        else:
            rgb_colors = texels  # (N, H, W, K, 3)
            alpha = torch.ones_like(
                rgb_colors[..., 0]
            )  # Set alpha to 1 for all RGB texels

        blended_rgb = blending.softmax_rgb_blend(
            rgb_colors, fragments, blend_params, znear=-256, zfar=256
        )

        blended_alpha = softmax_alpha_blend(alpha, fragments, blend_params)

        final_images = blended_rgb * blended_alpha[..., None] + background_color * (
            1 - blended_alpha[..., None]
        )

        return final_images


def render_normal(vertices, faces, camera_p):
    mesh = Meshes(
        verts=vertices,
        faces=faces,
    )

    vertex_normals = (
        mesh.verts_normals_padded()
    )  # Shape (B, V, 3) for batch size B and V vertices

    normals_as_colors = 0.5 * (vertex_normals + 1.0)  # Rescale to [0, 1]

    textures = TexturesVertex(verts_features=normals_as_colors)

    colored_mesh = Meshes(verts=vertices, faces=faces, textures=textures)

    cameras, height, width = create_perspective_camera(camera_p, vertices.device)

    raster_settings = RasterizationSettings(
        image_size=(height, width),  # (H, W)
        blur_radius=0.0,  # No blur
        faces_per_pixel=1,  # 1 face per pixel, non-blurry rendering
        bin_size=0,
    )

    blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb("white")) / 255.0)
    s = Shader(blend_params=blendparam)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=s,
    )

    imgs = renderer(colored_mesh, image_size=(height, width))

    return imgs


def render_result(vertices, faces, faces_uvs, uvs, texture_map, camera_p):
    mesh = create_mesh_with_texture(vertices, faces, uvs, faces_uvs, texture_map)
    cameras, height, width = create_perspective_camera(camera_p, vertices.device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),  # (H, W)
        blur_radius=np.log(1.0 / 1e-4) * 1e-7,
        faces_per_pixel=3,
        bin_size=-1,
    )
    blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb("white")) / 255.0)
    s = Shader(blend_params=blendparam)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=s,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=s,
    )

    imgs = renderer(mesh, image_size=(height, width))
    return imgs


def render_white_faces(mesh, cameras, raster_settings, img_size, light_p):
    device = mesh.device
    blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb("white")) / 255.0)  # noqa: F841
    lights = DirectionalLights(device=device, direction=((1, -1, 1),))
    materials = Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device,
    )
    s = SoftPhongShader(
        device=device, cameras=cameras, lights=lights, materials=materials
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=s,
    )
    return renderer(mesh, image_size=img_size)


def compute_edge_mask_from_pix_to_face(fragments, k=0):
    """
    Compute 1-pixel-wide triangle edge mask using face index jumps.
    Only uses the top-k fragment (default 0).
    """

    face_idx = fragments.pix_to_face[..., k]  # [B, H, W]
    B, H, W = face_idx.shape  # noqa: F841

    # Initialize edge mask
    edge_mask = torch.zeros_like(face_idx, dtype=torch.bool)

    # Compute horizontal edges (left-right)
    edge_x = torch.zeros_like(face_idx, dtype=torch.bool)
    edge_x[:, :, 1:] = face_idx[:, :, 1:] != face_idx[:, :, :-1]

    # Compute vertical edges (top-bottom)
    edge_y = torch.zeros_like(face_idx, dtype=torch.bool)
    edge_y[:, 1:, :] = face_idx[:, 1:, :] != face_idx[:, :-1, :]

    # Combine
    edge_mask = edge_x | edge_y  # [B, H, W]

    return edge_mask  # shape: [B, H, W]


def render_wireframe(
    mesh, face_mask, cameras, raster_settings, img_size, edge_color=(0, 0, 0)
):
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    class WireframeShader(torch.nn.Module):
        def __init__(self, blend_params=None, face_mask=None):
            super().__init__()
            self.face_mask = face_mask
            self.edge_color = torch.tensor(edge_color, dtype=torch.float32).cuda()
            self.blend_params = (
                blend_params if blend_params is not None else BlendParams()
            )

        def forward(self, fragments, meshes, **kwargs):
            # color to red
            pix_to_face = fragments.pix_to_face
            face_is_in_mask = torch.isin(pix_to_face, self.face_mask)
            red_color = torch.tensor([1, 0.6, 0.6], device=meshes.device)
            texels = meshes.sample_textures(fragments)
            red_texels = red_color.view(1, 1, 1, 1, 3).expand_as(texels)
            texels_colored = torch.where(
                face_is_in_mask.unsqueeze(-1), red_texels, texels
            )
            is_edge = compute_edge_mask_from_pix_to_face(fragments, k=0).unsqueeze(
                -1
            )  # [B, H, W]

            # Set the edges to black using the mask `is_edge`
            edge_color = self.edge_color.view(
                1, 1, 1, 3
            )  # (1, 1, 1, 3) -> Black color for edges
            edge_colors = edge_color.expand(
                fragments.pix_to_face.shape[0],
                fragments.pix_to_face.shape[1],
                fragments.pix_to_face.shape[2],
                fragments.pix_to_face.shape[3],
                3,
            )

            # Mask out non-edge pixels (set these to transparent: black with alpha 0)
            mask = (
                fragments.pix_to_face >= 0
            )  # Make sure we're only applying color to valid (visible) faces
            final_colors = texels_colored.clone()
            final_colors[mask & is_edge] = edge_colors[mask & is_edge]

            # Apply hard blending function to color the edges (use alpha = 1 since edges are opaque)
            images = blending.hard_rgb_blend(final_colors, fragments, self.blend_params)
            return images

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=WireframeShader(blend_params=blend_params, face_mask=face_mask),
    )
    return renderer(mesh, image_size=img_size)


def render_white_faces_with_black_edges(vertices, faces, camera_p, mask=None):
    if mask is None:
        face_mask = torch.tensor([])
    device = vertices.device
    cameras, height, width = create_perspective_camera(camera_p, device, reso=1)
    # print(cameras, height, width)
    mesh_white = white_face_mesh(vertices, faces)

    img_size = (height, width)
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=1e-9,
        faces_per_pixel=2,
        perspective_correct=True,
    )
    light_p = torch.linalg.inv(camera_p.extr)[:3, 3]
    white_image = (
        render_white_faces(mesh_white, cameras, raster_settings, img_size, light_p)
        * 1.2
    ).clamp_max(1)
    edge_image = render_wireframe(
        mesh_white, face_mask, cameras, raster_settings, img_size
    )
    # print(edge_image.shape)
    return edge_image * white_image
