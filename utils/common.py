import torch
import torchvision.transforms as T
from easydict import EasyDict as edict

N_Neighbors = 100


def normalize(v):
    """Normalize the quaternions (n x 4 tensor)."""
    norm = torch.norm(v, dim=2, keepdim=True)
    return v / norm


def log_loss(writer, loss_dict, iteration):
    for key, item in loss_dict.items():
        writer.add_scalar(f"loss/{key}", item, iteration)


def log_image(writer, label, predict, iteration, name):
    # cat the label and predict image
    img = torch.cat([label, predict], dim=2)
    writer.add_image(name, img, iteration)


def log_properties(writer, properties, iter):
    for key, item in properties.items():
        writer.add_image(f"properties/{key}", item.permute(2, 0, 1), iter)


def build_rotation(r):
    norm = torch.sqrt(
        r[..., 0] * r[..., 0]
        + r[..., 1] * r[..., 1]
        + r[..., 2] * r[..., 2]
        + r[..., 3] * r[..., 3]
    )

    q = r / norm[..., None]

    R = torch.zeros((*q.shape[:-1], 3, 3), device="cuda")

    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat2normal(q):
    r = build_rotation(q)
    return r[:, :, 2]


def quat_multiply(q1, q2):
    """
    q1, q2: nx4 tensor
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def warp_image(mask, images, camera):
    height, width = mask.shape
    vertical = mask.any(1)
    horizontal = mask.any(0)
    w1, w2 = horizontal.nonzero()[[0, -1]].view(-1).tolist()
    h1, h2 = vertical.nonzero()[[0, -1]].view(-1).tolist()
    bbox = [
        [max(0, w1 - 100), min(w2 + 100, width)],
        [max(0, h1 - 100), min(h2 + 100, height)],
    ]
    size_ = 1024
    width_bbox = bbox[0][1] - bbox[0][0]
    height_bbox = bbox[1][1] - bbox[1][0]
    warp = torch.eye(3).cuda()
    warp[0, 2] = -bbox[0][0]
    warp[1, 2] = -bbox[1][0]

    max_dim = max(width_bbox, height_bbox)
    scale = size_ / max_dim if max_dim > size_ else 1
    new_height = int(height_bbox * scale)
    new_width = int(width_bbox * scale)

    warp[:2] *= scale
    resize_transform = T.Resize((new_height, new_width))

    cut_images = [
        image[bbox[1][0] : bbox[1][1], bbox[0][0] : bbox[0][1]] for image in images
    ]
    cut_images_w = [
        resize_transform(cut_image.permute(2, 0, 1)).permute(1, 2, 0)
        for cut_image in cut_images
    ]
    cut_mask = mask[bbox[1][0] : bbox[1][1], bbox[0][0] : bbox[0][1]]
    cut_mask_w = resize_transform(cut_mask[None])

    camera["intr"] = warp @ camera["intr"]
    camera["width"] = new_width
    camera["height"] = new_height
    return (cut_images_w, cut_mask_w[0])


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


def calculate_area(uvs):
    x1, y1 = uvs[:, 0].unbind(-1)
    x2, y2 = uvs[:, 1].unbind(-1)
    x3, y3 = uvs[:, 2].unbind(-1)
    area = 0.5 * torch.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area


def calculate_boundary(uvs):
    max_ = uvs.max(dim=1).values
    min_ = uvs.min(dim=1).values
    return torch.column_stack((min_.floor(), max_.ceil()))


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


def load_barycentric_coordinates(size_, target_size, vertices, uvs, faces_uvs, faces):
    pixels, barycentric_coordinates, indices, uvs_img = pixels_barycentric_coordinates(
        uvs[faces_uvs], size_
    )
    pixels_t, barycentric_coordinates_t, indices_t, uvs_img_t = (
        pixels_barycentric_coordinates(uvs[faces_uvs], target_size, edge=False)
    )
    edge_neighbors, vertex_neighbors = calculate_neighbours(faces)
    return edict(
        {
            "pixels": pixels.long(),
            "barycentrics": barycentric_coordinates[..., None],
            "indices": faces[indices],
            "vertices": vertices,
            "face_uv": faces_uvs,
            "uv_verts": uvs,
            "uvs": uvs_img,
            "face_indices": indices,
            "face": faces,
            "neighbours": edge_neighbors,
            "pixels_t": pixels_t.long(),
            "barycentrics_t": barycentric_coordinates_t[..., None],
            "indices_t": faces[indices_t],
            "uvs_t": uvs_img_t,
            "face_indices_t": indices_t,
            "face_t": faces,
            "vertex_neighbors": vertex_neighbors,
        }
    )


def calculate_neighbours(faces):
    num_faces = faces.shape[0]

    edge_to_faces = {}
    print("calculating neighbors")
    for fi in range(num_faces):
        face = faces[fi].tolist()
        # For a triangle, the three edges are:
        edges = [
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[0], face[2]))),
            tuple(sorted((face[0], face[1]))),
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
        edge_neighbors = -torch.ones((num_faces, 3), dtype=torch.long)
    for fi in range(num_faces):
        face = faces[fi].tolist()
        edges = [
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[0], face[2]))),
            tuple(sorted((face[0], face[1]))),
        ]
        for edge_idx, edge in enumerate(edges):
            face_list = edge_to_faces.get(edge, [])
            neighbor = fi  # default if no neighbor is found
            for f_other in face_list:
                if f_other != fi:
                    neighbor = f_other
                    break
            edge_neighbors[fi, edge_idx] = neighbor

    vertex_to_faces = {}
    for fi in range(num_faces):
        face = faces[fi].tolist()
        for vertex in face:
            if vertex not in vertex_to_faces:
                vertex_to_faces[vertex] = []
            vertex_to_faces[vertex].append(fi)

    distance = 3
    vertex_neighbors_list = []
    face_to_neighbors = []
    for fi in range(num_faces):
        face = faces[fi].tolist()
        neighbor_set = set()
        for vertex in face:
            neighbor_set.update(vertex_to_faces[vertex])
        # Exclude the current face itself
        neighbor_set.discard(fi)
        face_to_neighbors.append(neighbor_set)

    for fi in range(num_faces):
        visited = set([fi])
        frontier = [fi]
        neighbor_list = []
        for d in range(distance):
            next_frontier = []
            for f in frontier:
                curr_neighbors = face_to_neighbors[f]
                for c in curr_neighbors:
                    if c not in visited:
                        visited.add(c)
                        next_frontier.append(c)
                        neighbor_list.append(c)
            frontier = next_frontier

        if len(neighbor_list) < N_Neighbors:
            neighbor_list.extend([-1] * (N_Neighbors - len(neighbor_list)))
        else:
            neighbor_list = neighbor_list[:N_Neighbors]
        vertex_neighbors_list.append(neighbor_list)

    vertex_neighbors_tensor = torch.tensor(vertex_neighbors_list, dtype=torch.long)

    return edge_neighbors, vertex_neighbors_tensor
