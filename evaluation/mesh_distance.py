
import trimesh
import torch
import numpy as np
import random
from chamfer_distance import ChamferDistance
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.options import load_yaml_options
import argparse

torch.manual_seed(2025)
np.random.seed(2025)

from geomloss import SamplesLoss  # Sinkhorn-based EMD

def parse_args():
    parser = argparse.ArgumentParser(description="mesh distance evaluation")
    parser.add_argument("-o", "--option", type=str, required=True, help="Path to options YAML file")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-r", "--rootpath", type=str)
    parser.add_argument("-g", "--group", type=str)
    parser.add_argument("-s", "--scene", type=str)
    parser.add_argument("-e", "--exp", type=str)
    return load_yaml_options(parser.parse_args())

def emd_approx_cuda(verts1: torch.Tensor, verts2: torch.Tensor, blur=0.001):
    verts1 = verts1.squeeze(0)
    verts2 = verts2.squeeze(0)
    n = min(len(verts1), len(verts2))
    x = verts1[:n].float().cuda()
    y = verts2[:n].float().cuda()

    loss_fn = SamplesLoss("sinkhorn", p=1, blur=blur)
    emd = loss_fn(x, y)
    return emd*100


def sample_mesh(m, n):
    vpos, face_index = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda"), face_index


def apply_displacement_verts(mesh, displacement_map):
    displacement_map = torch.tensor(np.load(displacement_map)).cuda()
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
    
    # UV 必須
    if mesh.visual.uv is None:
        raise ValueError("Mesh must have UV coordinates.")

    uv = torch.tensor(np.asarray(mesh.visual.uv)).float().cuda()
    H, W = displacement_map.shape
    uv_index = (uv * torch.tensor([W-1, H-1]).cuda()).to(torch.int64)
    uv_index[:, 0] = torch.clamp(uv_index[:, 0], 0, W - 1)
    uv_index[:, 1] = torch.clamp(uv_index[:, 1], 0, H - 1)

    displacement = displacement_map[uv_index[:,1], uv_index[:,0]]
    normals = torch.tensor(mesh.vertex_normals).float().cuda()
    displaced_vertices = vertices + normals * displacement[:, None]

    # Return displaced mesh
    displaced_mesh = mesh.copy()
    displaced_mesh.vertices = displaced_vertices.cpu().numpy()
    return displaced_mesh

def apply_blender_rotation(mesh):
    rotation = trimesh.transformations.rotation_matrix(
        np.radians(90), [1, 0, 0]
    )
    mesh.apply_transform(rotation)
    return mesh

def tessellate_mesh(mesh, iterations=6):
    # Simple tessellation
    for _ in range(iterations):
        mesh = mesh.subdivide()
    return mesh

def measure_chamfer_distance(scene, mesh1_path, mesh2_path, displacement_map, n_samples=100000, scale_factor=10.0, iterations=2):
    # Load meshes
    mesh1 = trimesh.load(mesh1_path)
    if isinstance(mesh1, trimesh.Scene):
        mesh1 = trimesh.util.concatenate([g for g in mesh1.geometry.values()])
    mesh1 = apply_blender_rotation(mesh1)
    mesh2 = trimesh.load(mesh2_path)
    if iterations > 0:
        mesh2 = tessellate_mesh(mesh2, iterations=iterations)
    mesh2 = apply_displacement_verts(mesh2, displacement_map)
    if iterations > 0:
        mesh2.export(mesh2_path.replace(".obj", "_displaced3.obj"))
    # Scale meshes
    bbox_size = np.max(mesh1.vertices.max(axis=0) - mesh1.vertices.min(axis=0))
    scale = scale_factor / bbox_size

    # Sample points
    vpos_mesh1, _ = sample_mesh(mesh1, n_samples)
    vpos_mesh2, face_index = sample_mesh(mesh2, n_samples)
    # Chamfer distance
    chamfer_dist = ChamferDistance()
    p1 = vpos_mesh1[None, ...] * scale 
    p2 = vpos_mesh2[None, ...] * scale 
    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    loss = (torch.mean(dist1)).item()
    emd_loss = emd_approx_cuda(p1 / scale, p2 / scale)  # 返回 shape [1]
    emd_value = emd_loss.item()
    return loss*100, (torch.mean(dist2)).item()*100, emd_value
 
if __name__ == "__main__":
    opt = parse_args()
    scene = opt.scene
    exp = opt.exp
    mesh_gt_path = f"{opt.rootpath}/{scene}/model.obj"

    mesh_target_path = f"{opt.group}/{exp}/mesh.obj"
    displacement_map = f"{opt.group}/{exp}/displacement.npy"
    loss, loss2, emd = measure_chamfer_distance(scene, mesh_gt_path, mesh_target_path, displacement_map)

    print(f"CD: {loss} , {loss2}; SD: {emd}")