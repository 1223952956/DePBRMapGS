import torch
import numpy as np
import smplx
import heapq
from collections import defaultdict
import argparse
import os


def calculate_area(uvs):
    x1, y1 = uvs[:, 0].unbind(-1)
    x2, y2 = uvs[:, 1].unbind(-1)
    x3, y3 = uvs[:, 2].unbind(-1)
    area = 0.5 * torch.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area


def map_to_indices(minmax):
    length = (minmax[:, 2:] - minmax[:, :2]).long()
    boxes = length[:, 0] * length[:, 1]
    indices = torch.repeat_interleave(torch.arange(len(minmax)), boxes.long())
    repeated_sequential_indices = torch.cat([torch.arange(count) for count in boxes])
    pix_indices = torch.column_stack(
        (repeated_sequential_indices, repeated_sequential_indices)
    )
    length = length[indices]
    pix_indices[:, 1] = pix_indices[:, 1] // length[:, 0]
    pix_indices[:, 0] = pix_indices[:, 0] % length[:, 0]

    return indices, pix_indices


def calculate_boundary(uvs):
    max_ = uvs.max(dim=1).values
    min_ = uvs.min(dim=1).values
    return torch.column_stack((min_.floor(), max_.ceil()))


def cal_barycentric_coordinates(uvs, areas, pixels):
    A = uvs[:, 0, :]  # (N, 2) - First vertex of each triangle
    B = uvs[:, 1, :]  # (N, 2) - Second vertex of each triangle
    C = uvs[:, 2, :]  # (N, 2) - Third vertex of each triangle
    P = pixels  # (N, 2) - The points we're calculating barycentric for

    # Step 2: Calculate signed areas for the full triangle and sub-triangles
    def signed_area(v1, v2, v3):
        """Calculates twice the signed area of the triangle formed by (v1, v2, v3)."""
        return (
            v1[:, 0] * (v2[:, 1] - v3[:, 1])
            + v2[:, 0] * (v3[:, 1] - v1[:, 1])
            + v3[:, 0] * (v1[:, 1] - v2[:, 1])
        )

    # Sub-triangle areas
    lambda1_area = signed_area(P, B, C)  # PBC
    lambda2_area = signed_area(P, C, A)  # PCA
    lambda3_area = signed_area(P, A, B)  # PAB

    # Step 3: Calculate the barycentric coordinates
    lambda1 = lambda1_area / areas
    lambda2 = lambda2_area / areas
    lambda3 = lambda3_area / areas

    # Combine into a single tensor (N, 3)
    barycentric_coordinates = torch.stack([lambda1, lambda2, lambda3], dim=1)

    return barycentric_coordinates


def pixels_barycentric_coordinates(uvs, size_, edge=False):
    uvs_img = uvs * (size_ - 1)
    bbox = calculate_boundary(uvs_img)
    indices, pix_indices = map_to_indices(bbox)
    uvs_img_clone = uvs_img.clone()
    uvs_img = uvs_img_clone[indices]
    boundaries = bbox[indices]
    pixels = boundaries[:, :2] + pix_indices
    areas = calculate_area(uvs_img)
    barycentric_coordinates = cal_barycentric_coordinates(
        uvs_img, areas * 2, pixels + 0.5
    )
    in_triangle = (barycentric_coordinates.min(dim=1).values >= 0) & (
        barycentric_coordinates.max(dim=1).values <= 1
    )

    pixels = pixels[in_triangle].long()
    barycentric_coordinates = barycentric_coordinates[in_triangle]
    indices = indices[in_triangle]
    uvs_img = uvs_img[in_triangle]
    if not edge:
        return pixels, barycentric_coordinates, indices, uvs_img / size_
    pixel_map = torch.zeros(size_, size_).bool().cuda()
    indices_map = torch.zeros(size_, size_).long().cuda()
    pixel_coord = (
        torch.stack(
            torch.meshgrid(torch.arange(size_), torch.arange(size_), indexing="xy"),
            dim=-1,
        )
        .long()
        .cuda()
    )
    neighbors = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 1],
        [-1, -1],
        [1, -1],
        [-1, 1],
        [2, 0],
        [-2, 0],
        [0, 2],
        [0, -2],
        [2, 2],
        [-2, -2],
        [2, -2],
        [-2, 2],
        [2, 1],
        [-2, 1],
        [2, -1],
        [-2, -1],
        [1, 2],
        [-1, 2],
        [1, -2],
        [-1, -2],
    ]
    for a, b in neighbors:
        pixel_map[pixels[:, 1] + a, pixels[:, 0] + b] = True
        indices_map[pixels[:, 1] + a, pixels[:, 0] + b] = indices
    pixel_map[pixels[:, 1], pixels[:, 0]] = False
    other_pixels = pixel_coord[pixel_map]
    other_indices = indices_map[pixel_map]
    other_uv_img = uvs_img_clone[other_indices]
    other_area = calculate_area(other_uv_img)
    other_bary = cal_barycentric_coordinates(
        other_uv_img, other_area * 2, other_pixels.float() + 0.5
    )
    all_pixels = torch.cat([pixels, other_pixels], dim=0)
    all_indices = torch.cat([indices, other_indices], dim=0)
    all_bary = torch.cat([barycentric_coordinates, other_bary], dim=0)
    all_uv_img = torch.cat([uvs_img, other_uv_img], dim=0)
    return all_pixels, all_bary, all_indices, all_uv_img / size_
    # return pixels, barycentric_coordinates, indices, uvs_img / size_


class UnionFind:
    def __init__(self, size):
        self.parent = torch.arange(size, dtype=torch.long)

    def find(self, x):
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Union by root
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def find_shortest_edge(uv_face, uvs):
    """
    Find the shortest edge in a triangle.

    Parameters:
        uv_face (torch.Tensor): (3,) Indices of the UV face vertices.
        uvs (torch.Tensor): (P, 2) UV coordinates.

    Returns:
        tuple: Indices of the two UV vertices forming the shortest edge.
    """
    verts = uvs[uv_face]  # noqa: F841
    edges = [
        (uv_face[0].item(), uv_face[1].item()),
        (uv_face[0].item(), uv_face[2].item()),
        (uv_face[1].item(), uv_face[2].item()),
    ]
    distances = [torch.norm(uvs[edge[0]] - uvs[edge[1]]).item() for edge in edges]
    min_idx = distances.index(min(distances))
    return edges[min_idx]


def remove_small_triangles(
    vertices, faces, uvs, uv_faces, size_, min_pixels=3, area_threshold=0.5
):
    face_uvs = uvs[uv_faces]
    uv_to_vertex = torch.ones((len(uvs))).long() * -1
    for face, uv_face in zip(faces, uv_faces):
        for i in range(3):
            uv_to_vertex[uv_face[i]] = face[i]
    areas = calculate_area(face_uvs)
    degenerate = areas * size_ * size_ < area_threshold
    _, _, triangle_indices, _ = pixels_barycentric_coordinates(face_uvs, size_)
    pixel_counts = torch.bincount(triangle_indices, minlength=faces.shape[0])
    insufficient_pixels = pixel_counts < min_pixels
    bad_triangles = degenerate | insufficient_pixels
    bad_triangle_indices = bad_triangles.nonzero(as_tuple=False).squeeze()
    if bad_triangle_indices.numel() == 0:
        return torch.arange(vertices.shape[0], dtype=torch.long)
    vertex_uf = UnionFind(vertices.shape[0])
    uv_vertex_uf = UnionFind(uvs.shape[0])

    # Build adjacency: UV vertex to UV faces
    uv_vertex_to_faces = defaultdict(set)
    for face_idx, face in enumerate(uv_faces.tolist()):
        for v in face:
            uv_vertex_to_faces[v].add(face_idx)

    # bad_triangles_queue = deque(bad_triangle_indices.tolist())
    bad_triangles_queue = []
    for idx in bad_triangle_indices:
        area = areas[idx].item() if idx < len(areas) else float("inf")
        heapq.heappush(
            bad_triangles_queue, (area, idx.item())
        )  # Store (area, triangle index)

    processed_triangles = set()
    max_iterations = 10000
    iteration = 0
    updates = torch.ones(len(faces)).float() * (-1)
    good_indices = set()
    while bad_triangles_queue and iteration < max_iterations:
        # current_bad_triangle = bad_triangles_queue.popleft()
        area, current_bad_triangle = heapq.heappop(bad_triangles_queue)
        if current_bad_triangle in processed_triangles:
            continue  # Skip if already processed
        if (updates[current_bad_triangle] != -1) and (
            area != updates[current_bad_triangle]
        ):
            continue
        if current_bad_triangle in good_indices:
            continue
        processed_triangles.add(current_bad_triangle)
        iteration += 1
        if current_bad_triangle >= uv_faces.shape[0]:
            continue  # Out of bounds
        uv_face = uv_faces[current_bad_triangle]
        uv_a, uv_b = find_shortest_edge(uv_face, uvs)

        uvs[uv_a] = (uvs[uv_a] + uvs[uv_b]) / 2
        uv_vertex_uf.union(uv_a, uv_b)
        uv_faces[uv_faces == uv_b] = uv_a
        mesh_vertices_a = uv_to_vertex[uv_a]
        mesh_vertices_b = uv_to_vertex[uv_b]
        if (mesh_vertices_a != -1) and (mesh_vertices_b != -1):
            vertex_uf.union(mesh_vertices_a, mesh_vertices_b)
            affected_faces_indices = (
                (
                    (faces == mesh_vertices_a).any(dim=1)
                    ^ (faces == mesh_vertices_b).any(dim=1)
                )
                .nonzero(as_tuple=False)
                .squeeze()
            )
            affected_faces_indices_ab = (
                (
                    (faces == mesh_vertices_a).any(dim=1)
                    & (faces == mesh_vertices_b).any(dim=1)
                )
                .nonzero(as_tuple=False)
                .squeeze()
            )
            if affected_faces_indices_ab.dim() == 0:
                affected_faces_indices_ab = affected_faces_indices_ab[None]
            for f in affected_faces_indices_ab.tolist():
                processed_triangles.add(f)

            affected_faces = uv_faces[affected_faces_indices]
            affected_uvs = uvs[affected_faces]
            affected_areas = calculate_area(affected_uvs)
            affected_degenerate = affected_areas * size_ * size_ < area_threshold
            _, _, affected_triangle_indices, _ = pixels_barycentric_coordinates(
                affected_uvs, size_
            )
            affect_pixel_counts = torch.bincount(
                affected_triangle_indices, minlength=affected_faces.shape[0]
            )
            affect_insufficient_pixels = affect_pixel_counts < min_pixels
            affect_bad_triangles = affected_degenerate | affect_insufficient_pixels
            affected_bad_indices = affected_faces_indices[affect_bad_triangles]
            affected_good_indices = affected_faces_indices[~affect_bad_triangles]
            for a, inx in zip(
                affected_areas[affect_bad_triangles].tolist(),
                affected_bad_indices.tolist(),
            ):
                if inx not in processed_triangles:
                    heapq.heappush(bad_triangles_queue, (a, inx))
                    updates[inx] = a
                if inx in good_indices:
                    good_indices.remove(inx)
            for inx in affected_good_indices.tolist():
                good_indices.add(inx)

            vertices[mesh_vertices_a] = (
                vertices[mesh_vertices_a] + vertices[mesh_vertices_b]
            ) / 2
            faces[faces == mesh_vertices_b] = mesh_vertices_a
            uv_to_vertex[uv_to_vertex == mesh_vertices_b] = mesh_vertices_a.clone()

    unique_roots, inverse_indices = torch.unique(faces, return_inverse=True)
    new_vertices = vertices[unique_roots]

    toremove = list(processed_triangles)
    index_bool = torch.ones(len(faces)).bool()
    index_bool[toremove] = False
    new_faces = inverse_indices[index_bool]

    unique_roots_uv, inverse_indices_uv = torch.unique(uv_faces, return_inverse=True)
    new_uv_faces = inverse_indices_uv[index_bool]
    new_uvs = uvs[unique_roots_uv]

    return new_vertices, new_faces, new_uvs, new_uv_faces


def load_mesh(mesh_path):
    vertices = []
    faces = []
    faces_uvs = []
    uvs = []
    with open(mesh_path, "r") as f:
        for line in f:
            if line.startswith("vt"):
                uvs.append([float(x) for x in line.split()[1:]])
            elif line.startswith("vn"):
                continue
            elif line.startswith("v"):
                vertices.append([float(x) for x in line.split()[1:]])
            elif line.startswith("f"):
                faces.append([int(x.split("/")[0]) for x in line.split()[1:]])
                faces_uvs.append([int(x.split("/")[1]) for x in line.split()[1:]])

    vertices = torch.tensor(vertices, dtype=torch.float).cuda()
    faces = torch.tensor(faces, dtype=torch.long).cuda() - 1
    faces_uvs = torch.tensor(faces_uvs, dtype=torch.long).cuda() - 1
    uvs = torch.tensor(uvs, dtype=torch.float).cuda()
    return vertices, faces, faces_uvs, uvs


def save_mesh(mesh_path, vertices, faces, faces_uvs, uvs):
    with open(mesh_path, "w") as f:
        for vertex in vertices:
            f.write(f"v {' '.join([str(x) for x in vertex.tolist()])}\n")
        for uv in uvs:
            f.write(f"vt {' '.join([str(x) for x in uv.tolist()])}\n")
        for face, face_uv in zip(faces, faces_uvs):
            f.write(
                f"f {' '.join([f'{x + 1}/{y + 1}' for x, y in zip(face.tolist(), face_uv.tolist())])}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate OBJ files from SMPLX parameters."
    )
    parser.add_argument(
        "-s", "--scene", type=str, default="Actor07", help="Scene name to process."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/data/actorshq/",
        help="Directory containing SMPLX parameters.",
    )
    parser.add_argument(
        "-i", "--pose_idx", type=int, default=0, help="Pose index to process."
    )
    return parser.parse_args()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    smpl_model = smplx.SMPLX(
        model_path=os.path.join(base_dir, "models/smplx"),
        mapping_list=os.path.join(base_dir, "pre-defined/mapping_list.npy"),
        gender="neutral",
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    ).cuda()
    torch.set_default_device("cuda")
    opt = parse_args()

    data_dir = f"{opt.data_dir}/{opt.scene}/Sequence1"
    smpl_data = np.load(f"{data_dir}/smpl_params.npz")
    smpl_data = {
        k: torch.from_numpy(v.astype(np.float32)).cuda() for k, v in smpl_data.items()
    }
    betas = smpl_data["betas"][0][None]

    global_orient = smpl_data["global_orient"][opt.pose_idx][None]
    transl = smpl_data["transl"][opt.pose_idx][None]
    expre = smpl_data["expression"][opt.pose_idx][None]
    jaw = smpl_data["jaw_pose"][opt.pose_idx][None]
    body_pose = smpl_data["body_pose"][opt.pose_idx][None]
    left_hand_pose = smpl_data["left_hand_pose"][opt.pose_idx][None]
    right_hand_pose = smpl_data["right_hand_pose"][opt.pose_idx][None]

    with torch.no_grad():
        live_smpl = smpl_model.forward(
            betas=betas,
            global_orient=global_orient,
            transl=transl,
            body_pose=body_pose,
            # jaw_pose = jaw,
            # expression = expre,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        vertices = live_smpl.vertices
        indices = torch.from_numpy(
            np.load(os.path.join(base_dir, "pre-defined/indices_no_eye.npy"))
        ).long()
        uvs = []
        faces = []
        with open(
            os.path.join(base_dir, "pre-defined/CL_no_eye_clear_Flat.obj"), "r"
        ) as f:
            for line in f:
                if line.startswith("vt"):
                    uvs.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith("f"):
                    faces.append(
                        list(
                            map(
                                lambda x: int(x.split("/")[1]), line.strip().split()[1:]
                            )
                        )
                    )
        v_out = vertices[0].tolist()

        output_dir = f"{opt.data_dir}/{opt.scene}/Sequence1/objs/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/{opt.pose_idx:08}-b.obj", "w") as file:
            for v in v_out:  # write vertices coordinates to file
                file.write("v {0} {1} {2}\n".format(v[0], v[1], v[2]))

            for uv in uvs:  # write uv mapping to file
                file.write("vt {0} {1}\n".format(uv[0], uv[1]))

            for f in faces:  # write faces to file
                file.write(
                    f"f {indices[f[0] - 1] + 1}/{f[0]} {indices[f[1] - 1] + 1}/{f[1]} {indices[f[2] - 1] + 1}/{f[2]}\n"
                )
        vertices, faces, faces_uvs, uvs = load_mesh(
            f"{output_dir}/{opt.pose_idx:08}-b.obj"
        )
        new_vertices, new_faces, new_uvs, new_uv_faces = remove_small_triangles(
            vertices, faces, uvs, faces_uvs, 1024
        )
        save_mesh(
            f"{output_dir}/{opt.pose_idx:08}.obj",
            new_vertices,
            new_faces,
            new_uv_faces,
            new_uvs,
        )
