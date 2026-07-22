import torch
from utils.general import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from pytorch3d.ops import knn_points
from utils.common import quat_multiply, N_Neighbors
from utils.sh import eval_sh
from utils.laplacian import compute_matrix, from_differential, to_differential
from matplotlib import pyplot as plt
import gaussian_splat_mesh
from utils.sh import RGB2SH


class OptimizationParams:
    def __init__(self):
        self.iterations = 5000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 5000
        self.feature_lr = 0.005
        self.opacity_lr = 0.01
        self.scaling_lr = 0.0005
        self.rotation_lr = 0.001
        self.percent_dense = 0.0005
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002


class GaussianModelBase:
    def __init__(self, opt, load_dir, mesh, train=True) -> None:
        self.optim_params = OptimizationParams()
        self.active_sh_degree = 0
        self.max_sh_degree = 3
        self.appearance_mode = getattr(
            opt,
            "appearance_mode",
            "sh" if getattr(opt, "sh", False) else "rgb",
        )
        self.sh = self.appearance_mode == "sh"
        self.spatial_lr_scale = 1
        self.percent_dense = 0.005
        self.indices = mesh.indices
        self.face_verts = mesh.face

        if "face_uv" in mesh.__dict__:
            self.face_uv = mesh.face_uv
        if "uv_verts" in mesh.__dict__:
            self.uv_verts = mesh.uv_verts
        self.barycentrics = mesh.barycentrics
        self.pixels = mesh.pixels
        self.uvs = mesh.uvs
        self.curr_bary = mesh.barycentrics.clone()[:, :2]
        if "vertices" in mesh.__dict__:
            self.original_verts = mesh.vertices.clone()
        self.curr_pixels = mesh.pixels.clone()
        self.pos_lr = opt.pos_scale
        self.count = len(self.indices)
        self.threed = False

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @property
    def get_vertices(self):
        v = from_differential(self.M, self._u)
        return v

    def get_face_normals(self, verts):
        edge1 = verts[:, 1] - verts[:, 0]
        edge2 = verts[:, 2] - verts[:, 0]
        cross = torch.cross(edge1, edge2, dim=1)
        normals = torch.nn.functional.normalize(cross, dim=1)
        return normals

    def get_xyz(self, vertices, normals):
        self.original_vertices = vertices.detach().clone()
        self.original_barycentric = self.curr_bary.detach().clone()
        self.original_normal = normals.detach().clone()
        offset = self._offset
        points_mean = (vertices[:, :2] * self.curr_bary).sum(1) + vertices[:, 2] * (
            1 - self.curr_bary.sum(1)
        )
        return normals * offset + points_mean

    def cal_scale(self, factor=1, init=False):
        verts = (self.get_vertices)[self.indices]
        normals = self.get_face_normals(verts)
        points = self.get_xyz(verts, normals).detach()
        dist2 = torch.clamp_min(
            knn_points(points[None], points[None], K=4)[0][0, :, 1:].mean(-1), 0.0000001
        )
        if init:
            self.scales = torch.log(torch.sqrt(dist2) / factor)[..., None].repeat(1, 3)
            self.init_scales = self.scales.clone()
        else:
            self.scales.data = torch.log(torch.sqrt(dist2) / factor)[..., None].repeat(
                1, 3
            )

    def get_gaussian_vals(self, t=None, camera=None):
        verts = self.get_vertices
        mean = verts[self.indices]
        normals = self.get_face_normals(mean)
        res = {
            "verts": verts,
            "positions": self.get_xyz(mean, normals),
            "opacity": self.get_opacity,
            "scales": self.get_scaling,
            "rotations": self.get_rotation(normals),
            "max_sh_degree": self.active_sh_degree,
        }
        if self.appearance_mode == "pbr":
            res.update(
                {
                    "normals": normals,
                    "albedo": self.get_albedo,
                    "roughness": self.get_roughness,
                    "metallic": self.get_metallic,
                }
            )
        elif self.appearance_mode == "rgb":
            res["colors"] = self.get_colors()
        elif self.appearance_mode == "sh":
            res["shs"] = self.get_colors()
        else:
            raise ValueError(f"Unsupported appearance mode: {self.appearance_mode}")
        return res

    def get_rotation(self, normals):
        rotation = torch.nn.functional.normalize(self._rotation, dim=1)
        normal_quat = torch.column_stack(
            (
                normals[:, 2] + 1,
                -normals[:, 1],
                normals[:, 0],
                torch.zeros_like(normals[:, 0]),
            )
        )
        normal_quat = torch.nn.functional.normalize(normal_quat)
        surface_quat = quat_multiply(rotation, normal_quat)
        return surface_quat

    def get_colors(self, xyzs=None, camera=None):
        if self.appearance_mode == "sh":
            return torch.cat((self._features_dc, self._features_rest), dim=1)
        if self.appearance_mode == "rgb":
            return torch.sigmoid(self._colors)
        raise ValueError(
            f"get_colors() is not available in {self.appearance_mode} mode"
        )

    @property
    def get_albedo(self):
        return torch.sigmoid(self._albedo)

    @property
    def get_roughness(self):
        return 0.04 + 0.96 * torch.sigmoid(self._roughness)

    @property
    def get_metallic(self):
        return torch.sigmoid(self._metallic)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        if self.threed:
            return torch.exp(self.scales)
        else:
            return torch.exp(self.scales[..., :2])


class GaussianModelVis(GaussianModelBase):
    def __init__(self, opt, load_dir, mesh, train=True):
        super().__init__(opt, load_dir, mesh, train)
        self.opt = opt
        self.vis = True
        self.restore(load_dir, opt.other_path)

    @property
    def get_vertices(self):
        return self.vertices

    def restore(self, load_dir, arg):
        print("--- restore Gaussian: ", load_dir, "---")
        checkpoint = torch.load(load_dir)

        if isinstance(checkpoint, dict):
            self._restore_dict_checkpoint(checkpoint)
        elif isinstance(checkpoint, (tuple, list)):
            self._restore_legacy_checkpoint(checkpoint)
        else:
            raise TypeError(
                "Unsupported Gaussian checkpoint type: "
                f"{type(checkpoint).__name__}."
            )

        self._validate_appearance_parameters()
        self._freeze_parameters()

    def _restore_dict_checkpoint(self, checkpoint):
        self.appearance_mode = checkpoint.get(
            "appearance_mode", self.appearance_mode
        )
        self.sh = self.appearance_mode == "sh"

        self.vertices = checkpoint["vertices"]
        self._offset = checkpoint["offset"]
        self._colors = checkpoint.get("colors")
        self._features_dc = checkpoint.get("features_dc")
        self._features_rest = checkpoint.get("features_rest")
        self.scales = checkpoint["scales"]
        self._opacity = checkpoint["opacity"]
        self._rotation = checkpoint["rotation"]
        self.faces = checkpoint["faces"]
        self.indices = checkpoint["indices"]
        self.curr_bary = checkpoint["curr_bary"]
        self._albedo = checkpoint.get("albedo")
        self._roughness = checkpoint.get("roughness")
        self._metallic = checkpoint.get("metallic")

    def _restore_legacy_checkpoint(self, checkpoint):
        if len(checkpoint) != 11:
            raise ValueError(
                "Unsupported legacy Gaussian checkpoint with "
                f"{len(checkpoint)} entries; expected 11."
            )

        (
            self.vertices,
            self._offset,
            self._colors,
            self._features_dc,
            self._features_rest,
            self.scales,
            self._opacity,
            self._rotation,
            self.faces,
            self.indices,
            self.curr_bary,
        ) = checkpoint
        self._albedo = None
        self._roughness = None
        self._metallic = None

    def _validate_appearance_parameters(self):
        if self.appearance_mode == "pbr":
            missing = [
                name
                for name, value in (
                    ("albedo", self._albedo),
                    ("roughness", self._roughness),
                    ("metallic", self._metallic),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "PBR checkpoint is missing required parameters: "
                    + ", ".join(missing)
                )

    def _freeze_parameters(self):
        for tensor in (
            self.vertices,
            self._offset,
            self._colors,
            self._features_dc,
            self._features_rest,
            self.scales,
            self._opacity,
            self._rotation,
            self._albedo,
            self._roughness,
            self._metallic,
        ):
            if tensor is not None:
                tensor.requires_grad_(False)


class GaussianModelEditColor(GaussianModelVis):
    @torch.no_grad()
    def change_color(self, color_pic):
        uvs = self.uv_verts[self.face_uv]
        size_h, size_w = color_pic.shape[:2]

        uvs[..., 0] *= size_w - 1
        uvs[..., 1] *= size_h - 1
        uvs = uvs[self.faces]
        target_uvs = (self.curr_bary * uvs[:, :2]).sum(1) + uvs[:, 2] * (
            1 - self.curr_bary.sum(1)
        )
        target_uvs = (target_uvs + 0.5).long()
        target_color = color_pic[target_uvs[:, 1], target_uvs[:, 0]]
        target_alpha = target_color[..., [3]]
        self._colors = (inverse_sigmoid(target_color[..., :3])).clamp_min(
            -10
        ) * target_alpha + (1 - target_alpha) * self.original_colors

    def restore(self, load_dir, color_path):
        print("--- restore Gaussian: ", load_dir, "---")
        super().restore(load_dir, color_path)
        self.original_colors = self._colors.clone()
        load_color = torch.from_numpy(plt.imread(color_path)).cuda().float()
        self.load_color = load_color
        self.change_color(load_color)


class GaussianModelTrain(GaussianModelBase):
    def __init__(self, opt, load_dir, mesh, train=True, vertices=None) -> None:
        super().__init__(opt, load_dir, mesh, train)
        self.sh = self.appearance_mode == "sh"
        self._lambda = opt.lambda_initial
        vertices = mesh.vertices

        self.vertex_neighbors = mesh.vertex_neighbors
        self.neighbor = mesh.neighbours
        self.curr_bary = mesh.barycentrics[:, :2]
        mean = vertices.mean(dim=0)
        vertices = vertices - mean[None]
        self.faces = mesh.face_indices
        self.eye, self.L = compute_matrix(opt, vertices, mesh.indices)
        self.M = torch.add(self.eye, self._lambda * self.L).coalesce()
        if self._lambda == 0:
            self.M = self.eye.coalesce()
        self._t = nn.Parameter(mean[None].requires_grad_(True))
        self._r = nn.Parameter(
            torch.tensor([1, 0, 0, 0]).float().cuda().requires_grad_(True)
        )
        self._s = nn.Parameter(torch.ones((1, 3)).float().cuda().requires_grad_(True))
        if self._lambda == 0:
            self._u = nn.Parameter(vertices.requires_grad_(True))
        else:
            self._u = nn.Parameter(
                to_differential(self.M, vertices).requires_grad_(True)
            )
        self.count = len(self.indices)
        self._offset = nn.Parameter(
            torch.zeros((self.count, 1)).float().cuda().requires_grad_(True)
        )
        self._colors = nn.Parameter(
            inverse_sigmoid(
                torch.ones(self.count, 3).float().cuda() - 0.5
            ).requires_grad_(self.appearance_mode == "rgb")
        )
        fused_color = RGB2SH(
            torch.tensor(torch.ones(self.count, 3) - 0.5).float().cuda()
        )

        self._albedo = nn.Parameter(
            inverse_sigmoid(
                torch.full((self.count, 3), 0.5, device="cuda")
            ).requires_grad_(self.appearance_mode == "pbr")
        )

        self._roughness = nn.Parameter(
            inverse_sigmoid(
                torch.full((self.count, 1), 0.5, device="cuda")
            ).requires_grad_(self.appearance_mode == "pbr")
        )

        self._metallic = nn.Parameter(
            inverse_sigmoid(
                torch.full((self.count, 1), 0.05, device="cuda")
            ).requires_grad_(self.appearance_mode == "pbr")
        )

        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        self._features_dc = nn.Parameter(
            features[:, :, 0:1]
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(self.appearance_mode == "sh")
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:]
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(self.appearance_mode == "sh")
        )
        self._opacity = nn.Parameter(
            inverse_sigmoid(
                0.95 * torch.ones((self.count, 1)).float().cuda()
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor([[1, 0, 0, 0]])
            .tile(self.count, 1)
            .float()
            .cuda()
            .requires_grad_(True)
        )
        self.cal_scale(init=True)
        self.xyz_gradient_accum = torch.zeros((self.count, 1), device="cuda")
        self.denom = torch.zeros((self.count, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.count), device="cuda")
        self.pixels_t = mesh.pixels_t
        self.indices_t = mesh.indices_t
        self.barycentrics_t = mesh.barycentrics_t
        self.uvs_t = mesh.uvs_t
        self.face_indices_t = mesh.face_indices_t
        self.face_t = mesh.face_t
        self.tmp_radii = None

    @property
    def get_vertices(self):
        if self._lambda == 0:
            v = self._u
        else:
            v = from_differential(self.M, self._u)
        return (build_rotation(self._r[None])[0] @ v.T).T * self._s + self._t

    @property
    def get_scaling(self):
        if self.threed:
            return torch.exp(self.scales)
        else:
            return torch.exp(self.scales[..., :2])

    @torch.no_grad()
    def revise_vertices(self):
        if self._lambda == 0:
            return
        vertices0 = from_differential(self.M, self._u)
        vertices = vertices0[self.indices]
        original_normal = (
            self.original_normal @ build_rotation(self._r[None])[0] / self._s
        )
        points_mean = (vertices[:, :2] * self.original_barycentric).sum(1) + vertices[
            :, 2
        ] * (1 - self.original_barycentric.sum(1))
        target_points = points_mean.detach() + self._offset.detach() * original_normal
        device = vertices.device
        barycentric = torch.cat(
            (
                self.original_barycentric,
                1.0 - self.original_barycentric.sum(dim=1, keepdim=True),
            ),
            dim=1,
        )
        a = barycentric * target_points[:, None]
        b = barycentric[..., 0]
        vertex_sum = vertices0.clone()
        vertex_weight = torch.ones((vertices0.shape[0]), device=device)
        vertex_sum.scatter_add_(
            0, self.indices[:, :, None].expand(-1, -1, 3).view(-1, 3), a.view(-1, 3)
        )
        vertex_weight.scatter_add_(0, self.indices.view(-1), b.view(-1))
        new_vertices = vertex_sum / (vertex_weight[:, None])
        diff = -new_vertices + vertices0
        u_diff = from_differential(self.M, diff)
        self._u.data = self._u.data - u_diff

    def averate_offset(self):
        face_offset_sum = torch.zeros(
            (self.face_verts.shape[0], 1), device=self._offset.device
        )
        face_offset_count = torch.zeros(
            (self.face_verts.shape[0], 1), device=self._offset.device
        )
        face_offset_abs = torch.zeros(
            (self.face_verts.shape[0], 1), device=self._offset.device
        )
        face_offset_sum.scatter_add_(0, self.faces[:, None], self._offset)
        face_offset_abs.scatter_add_(0, self.faces[:, None], torch.abs(self._offset))
        face_offset_count.scatter_add_(
            0, self.faces[:, None], torch.ones_like(self._offset)
        )

        face_offset_avg = face_offset_sum / face_offset_count.clamp_min(1e-6)
        self.unhealthy_filter = face_offset_count.squeeze(1) < 1
        face_offset_abs = face_offset_abs / face_offset_count.clamp_min(1e-6)
        return face_offset_avg, face_offset_abs, face_offset_count

    def attach_splat(self, face_mask):
        face_mask = face_mask.squeeze()
        new_indices = face_mask.nonzero(as_tuple=False).squeeze(1)
        if len(new_indices) == 0:
            return
        new_faces = self.face_verts[face_mask]
        device = self._offset.device
        new_bary = 0.33 * torch.ones((new_faces.shape[0], 2, 1), device=device)
        new_rotation = (
            torch.tensor([[1, 0, 0, 0]]).tile(new_faces.shape[0], 1).float().cuda()
        )
        new_offset = torch.zeros((new_faces.shape[0], 1), device=device)
        new_color = torch.rand((new_faces.shape[0], 3), device=device)
        new_features_dc, new_features_rest = None, None
        new_albedo, new_roughness, new_metallic = None, None, None
        if self.appearance_mode == "sh":
            new_features_dc = RGB2SH(new_color)[:, :3, 0:1]
            new_features_rest = torch.zeros(
                (new_faces.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1),
                device=device,
            )
        if self.appearance_mode == "pbr":
            new_albedo = inverse_sigmoid(
                torch.full((new_faces.shape[0], 3), 0.5, device=device)
            )
            new_roughness = inverse_sigmoid(
                torch.full((new_faces.shape[0], 1), 0.5, device=device)
            )
            new_metallic = inverse_sigmoid(
                torch.full((new_faces.shape[0], 1), 0.05, device=device)
            )
        new_opacity = inverse_sigmoid(
            torch.ones((new_faces.shape[0], 1), device=device) * 0.9
        )
        new_tmp_radii = torch.zeros((new_faces.shape[0]), device=device)

        self.indices = torch.cat((self.indices, new_faces), dim=0)
        self.faces = torch.cat((self.faces, new_indices), dim=0)

        verts = self.get_vertices[new_faces]
        areas = self.calculate_face_area(verts)
        raduis = torch.sqrt(areas / torch.pi)
        new_scaling = torch.log(raduis / 1.5)[..., None].repeat(1, 3)

        self.densification_postfix(
            new_offset,
            new_color,
            new_opacity,
            new_bary,
            new_scaling,
            new_rotation,
            new_tmp_radii,
            new_features_dc,
            new_features_rest,
            new_albedo,
            new_roughness,
            new_metallic,
        )

    def calculate_face_area(self, vertices=None):
        if vertices is None:
            vertices = self.get_vertices[self.face_verts]
        edge1 = vertices[:, 1] - vertices[:, 0]
        edge2 = vertices[:, 2] - vertices[:, 0]
        cross = torch.cross(edge1, edge2, dim=1)
        areas = torch.norm(cross, dim=1) * 0.5
        return areas

    @torch.no_grad()
    def revise_offset(self):
        new_vertices = self.get_vertices[self.indices]
        verts_delta = new_vertices - self.original_vertices
        points_delta = (verts_delta[:, :2] * self.original_barycentric).sum(
            1
        ) + verts_delta[:, 2] * (1 - self.original_barycentric.sum(1))
        offset_revise = (self.original_normal * points_delta).sum(1)
        self._offset.data = self._offset.data - offset_revise[:, None] * 0.1

    @torch.no_grad()
    def capture(self, name):
        checkpoint = {
            "format_version": 2,
            "appearance_mode": self.appearance_mode,

            "vertices": self.get_vertices,
            "offset": self._offset,
            "colors": self._colors if self.appearance_mode == "rgb" else None,
            "features_dc": (
                self._features_dc if self.appearance_mode == "sh" else None
            ),
            "features_rest": (
                self._features_rest if self.appearance_mode == "sh" else None
            ),
            "scales": self.scales,
            "opacity": self._opacity,
            "rotation": self._rotation,
            "faces": self.faces,
            "indices": self.indices,
            "curr_bary": self.curr_bary,

            "albedo": self._albedo,
            "roughness": self._roughness,
            "metallic": self._metallic,
        }

        torch.save(checkpoint, name)

    @torch.no_grad()
    def update_triangle(self):
        curr_bary = self.curr_bary.data
        third = 1 - curr_bary.sum(dim=1)
        trouble1 = curr_bary[:, 0] < 0
        trouble2 = curr_bary[:, 1] < 0
        trouble3 = third < 0
        troubled_mask = trouble1 | trouble2 | trouble3
        troubled_indices = troubled_mask.squeeze(1).nonzero(as_tuple=False).squeeze(1)
        if troubled_indices.numel() == 0:
            return
        b = curr_bary[troubled_indices]
        b[b < 0] = 0
        new_b = torch.nn.functional.normalize(b, p=1, dim=1)
        curr_bary[troubled_indices] = new_b
        neighbor_i = torch.zeros_like(troubled_indices)
        trouble1_troubled = trouble1[troubled_indices].squeeze(1)
        trouble2_troubled = trouble2[troubled_indices].squeeze(1)
        trouble3_troubled = trouble3[troubled_indices].squeeze(1)
        neighbor_i[trouble1_troubled] = 0
        neighbor_i[trouble2_troubled] = 1
        neighbor_i[trouble3_troubled] = 2
        curr_face_idx = self.faces[troubled_indices]
        moved_to = self.neighbor[curr_face_idx, neighbor_i]
        self.indices[troubled_indices] = self.face_verts[moved_to]
        self.faces[troubled_indices] = moved_to
        bary_index = 5
        group = self.optimizer.param_groups[bary_index]
        stored_state = self.optimizer.state.get(group["params"][0], None)

        if stored_state is not None:
            stored_state["exp_avg"][troubled_indices] = 0
            stored_state["exp_avg_sq"][troubled_indices] = 0

            del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(curr_bary)
            self.optimizer.state[group["params"][0]] = stored_state
            self.curr_bary.data = group["params"][0]
        else:
            self.curr_bary = nn.Parameter(curr_bary)

    def training_setup(self):
        self.scales = nn.Parameter((self.scales))
        self.scales.requires_grad_(False)
        l = [  # noqa: E741
            {
                "params": [self._u],
                "lr": self.optim_params.position_lr_init * self.pos_lr,
                "name": "xyz",
            },
            {
                "params": [self._offset],
                "lr": self.optim_params.position_lr_final,
                "name": "offset",
            },
            {
                "params": [self._opacity],
                "lr": self.optim_params.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._rotation],
                "lr": self.optim_params.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self.scales],
                "lr": self.optim_params.scaling_lr,
                "name": "scaling",
            },
            {"params": [self.curr_bary], "lr": 0.0001, "name": "bary"},
        ]
        if self.appearance_mode == "sh":
            l.append(
                {
                    "params": [self._features_dc],
                    "lr": self.optim_params.feature_lr,
                    "name": "f_dc",
                }
            )
            l.append(
                {
                    "params": [self._features_rest],
                    "lr": self.optim_params.feature_lr / 20.0,
                    "name": "f_rest",
                }
            )
        elif self.appearance_mode == "rgb":
            l.append(
                {
                    "params": [self._colors],
                    "lr": self.optim_params.feature_lr,
                    "name": "f_dc",
                }
            )
        elif self.appearance_mode == "pbr":
            l.extend(
                [
                    {   "params": [self._albedo], 
                        "lr": 0.005, 
                        "name": "albedo",
                    },
                    {
                        "params": [self._roughness],
                        "lr": 0.001,
                        "name": "roughness",
                    },
                    {
                        "params": [self._metallic],
                        "lr": 0.001,
                        "name": "metallic",
                    },
                ]
            )

        self.optimizer = torch.optim.Adam(l, eps=1e-15)

        l_g = [
            {"params": [self._s], "lr": 0.1 * self.pos_lr, "name": "s"},
            {"params": [self._t], "lr": 0.001 * self.pos_lr, "name": "t"},
            {"params": [self._r], "lr": self.optim_params.rotation_lr, "name": "r"},
        ]
        self.optimizer_g = torch.optim.Adam(l_g, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=self.optim_params.position_lr_init,
            lr_final=self.optim_params.position_lr_final * self.pos_lr,
            lr_delay_mult=self.optim_params.position_lr_delay_mult,
            max_steps=self.optim_params.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def update_global_rate(self):
        for param_group in self.optimizer_g.param_groups:
            if param_group["name"] == "t":
                param_group["lr"] = 0.0001
            if param_group["name"] == "s":
                param_group["lr"] = 0.0001
            if param_group["name"] == "r":
                param_group["lr"] = 0.00001

    def add_densification_stats(self, viewspace_point_tensor, visibility_filter):
        self.xyz_gradient_accum[visibility_filter] += torch.norm(
            viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True
        )
        self.denom[visibility_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        count_before = int(self.indices.shape[0])
        valid_grad_mask = self.denom.squeeze(-1) > 0
        valid_grads = grads.squeeze(-1)[valid_grad_mask]
        if valid_grads.numel() > 0:
            grad_mean = float(valid_grads.mean().item())
            grad_p99 = float(torch.quantile(valid_grads, 0.99).item())
            grad_max = float(valid_grads.max().item())
        else:
            grad_mean = 0.0
            grad_p99 = 0.0
            grad_max = 0.0

        self.tmp_radii = radii
        clone_count = self.densify_and_clone(grads, max_grad, extent)
        split_parent_count = self.densify_and_split(grads, max_grad, extent)
        count_after_densify = int(self.indices.shape[0])

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        prune_count = int(prune_mask.sum().item())
        self.prune_points(prune_mask)
        count_after = int(self.indices.shape[0])
        self.tmp_radii = None

        torch.cuda.empty_cache()

        return {
            "count_before": count_before,
            "valid_grad_count": int(valid_grads.numel()),
            "grad_mean": grad_mean,
            "grad_p99": grad_p99,
            "grad_max": grad_max,
            "grad_threshold": float(max_grad),
            "clone_count": clone_count,
            "split_parent_count": split_parent_count,
            "split_child_count": 2 * split_parent_count,
            "count_after_densify": count_after_densify,
            "prune_count": prune_count,
            "count_after": count_after,
            "net_change": count_after - count_before,
        }

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        self.indices = self.indices[valid_points_mask]
        self.faces = self.faces[valid_points_mask]
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._offset = optimizable_tensors["offset"]
        if self.appearance_mode == "rgb":
            self._colors = optimizable_tensors["f_dc"]
        elif self.appearance_mode == "sh":
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        elif self.appearance_mode == "pbr":
            self._albedo = optimizable_tensors["albedo"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]
        self._opacity = optimizable_tensors["opacity"]
        self.curr_bary = optimizable_tensors["bary"]
        self._rotation = optimizable_tensors["rotation"]
        self.scales = optimizable_tensors["scaling"]

        self.count = len(self.scales)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        target_group = ["offset", "opacity", "rotation", "bary", "scaling"]
        if self.appearance_mode == "rgb":
            target_group.append("f_dc")
        elif self.appearance_mode == "sh":
            target_group.extend(["f_dc", "f_rest"])
        elif self.appearance_mode == "pbr":
            target_group.extend(["albedo", "roughness", "metallic"])
        for group in self.optimizer.param_groups:
            if group["name"] not in target_group:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.indices.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        vertices = self.get_vertices[self.face_verts]
        mean_area = self.calculate_face_area(vertices).mean()
        _ = (mean_area / torch.pi).sqrt()
        selected_pts_mask = torch.logical_or(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        selected_count = int(selected_pts_mask.sum().item())

        stds = 0.05 * torch.ones((selected_pts_mask.sum(), 2), device="cuda")
        means = torch.zeros((stds.size(0), 2), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        new_scaling = torch.log(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        if not self.threed:
            scale3 = self.scales[selected_pts_mask][:, [2]].repeat(N, 1)
            new_scaling = torch.column_stack((new_scaling, scale3))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_bary = (self.curr_bary[selected_pts_mask] + samples.unsqueeze(-1)).repeat(
            N, 1, 1
        )
        new_color = None
        if self.appearance_mode == "rgb":
            new_color = self._colors[selected_pts_mask].repeat(N, 1)
        new_features_dc = None
        new_features_rest = None
        if self.appearance_mode == "sh":
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_albedo, new_roughness, new_metallic = None, None, None
        if self.appearance_mode == "pbr":
            new_albedo = self._albedo[selected_pts_mask].repeat(N, 1)
            new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
            new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)
        new_offset = self._offset[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_indices = self.indices[selected_pts_mask].repeat(N, 1)
        self.indices = torch.cat([self.indices, new_indices], dim=0)
        new_faces = self.faces[selected_pts_mask].repeat(N)
        self.faces = torch.cat([self.faces, new_faces], dim=0)
        self.densification_postfix(
            new_offset,
            new_color,
            new_opacity,
            new_bary,
            new_scaling,
            new_rotation,
            new_tmp_radii,
            new_features_dc,
            new_features_rest,
            new_albedo,
            new_roughness,
            new_metallic,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)
        return selected_count

    def densification_postfix(
        self,
        new_offset,
        new_color,
        new_opacity,
        new_bary,
        new_scaling,
        new_rotation,
        new_tmp_radii,
        new_features_dc=None,
        new_features_rest=None,
        new_albedo=None,
        new_roughness=None,
        new_metallic=None,
    ):
        d = {
            "offset": new_offset,
            "opacity": new_opacity,
            "rotation": new_rotation,
            "scaling": new_scaling,
            "bary": new_bary,
        }
        if self.appearance_mode == "rgb":
            d["f_dc"] = new_color
        elif self.appearance_mode == "sh":
            d.update(
                {
                    "f_dc": new_features_dc,
                    "f_rest": new_features_rest,
                }
            )
        elif self.appearance_mode == "pbr":
            d.update(
                {
                    "albedo": new_albedo,
                    "roughness": new_roughness,
                    "metallic": new_metallic,
                }
            )

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._offset = optimizable_tensors["offset"]
        if self.appearance_mode == "sh":
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        elif self.appearance_mode == "rgb":
            self._colors = optimizable_tensors["f_dc"]
        elif self.appearance_mode == "pbr":
            self._albedo = optimizable_tensors["albedo"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]
        self._opacity = optimizable_tensors["opacity"]
        self.curr_bary = optimizable_tensors["bary"]
        self._rotation = optimizable_tensors["rotation"]
        self.scales = optimizable_tensors["scaling"]
        self.count = len(self.scales)
        self.xyz_gradient_accum = torch.zeros((self.count, 1), device="cuda")
        self.denom = torch.zeros((self.count, 1), device="cuda")
        if self.tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.max_radii2D = torch.zeros((self._offset.shape[0]), device="cuda")

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        selected_count = int(selected_pts_mask.sum().item())

        new_scaling = self.scales[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_bary = self.curr_bary[selected_pts_mask]
        new_color = None
        if self.appearance_mode == "rgb":
            new_color = self._colors[selected_pts_mask]
        if self.appearance_mode == "sh":
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
        else:
            new_features_dc = None
            new_features_rest = None
        new_albedo, new_roughness, new_metallic = None, None, None
        if self.appearance_mode == "pbr":
            new_albedo = self._albedo[selected_pts_mask]
            new_roughness = self._roughness[selected_pts_mask]
            new_metallic = self._metallic[selected_pts_mask]
        new_offset = self._offset[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_indices = self.indices[selected_pts_mask]
        self.indices = torch.cat([self.indices, new_indices], dim=0)
        new_faces = self.faces[selected_pts_mask]
        self.faces = torch.cat([self.faces, new_faces], dim=0)
        self.densification_postfix(
            new_offset,
            new_color,
            new_opacity,
            new_bary,
            new_scaling,
            new_rotation,
            new_tmp_radii,
            new_features_dc,
            new_features_rest,
            new_albedo,
            new_roughness,
            new_metallic,
        )
        return selected_count

    def reset_opacity(self):
        op = self.get_opacity
        opacities_new = inverse_sigmoid(torch.min(op, torch.ones_like(op) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_offset(self):
        offset_new = torch.zeros_like(self._offset)
        optimizable_tensors = self.replace_tensor_to_optimizer(offset_new, "offset")
        self._offset = optimizable_tensors["offset"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def fill_map(self, map):
        res = torch.zeros_like(map)
        h, w = map.shape[:2]
        pixels = self.pixels_t.to(torch.int)
        neighbors = [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
            [2, 0],
            [-2, 0],
            [0, 2],
            [0, -2],
            [3, 0],
            [-3, 0],
            [0, 3],
            [0, -3],
            [4, 0],
            [-4, 0],
            [0, 4],
            [0, -4],
        ]
        for a, b in neighbors:
            res[pixels[:, 1] + a, pixels[:, 0] + b] = map[pixels[:, 1], pixels[:, 0]]
        res[pixels[:, 1], pixels[:, 0]] = map[pixels[:, 1], pixels[:, 0]]
        return res

    def project_property_to_uv(self, opt):
        reso = opt.target_size
        vertices = self.get_vertices
        normals_faces = self.get_face_normals(vertices[self.face_t])
        normals_all = self.get_face_normals(vertices[self.indices])
        xyzs_all = self.get_xyz(vertices[self.indices], normals_all)
        rotations_all = build_rotation(self.get_rotation(normals_all))
        scales_all = self.get_scaling[..., :2]
        opacities_all = self.get_opacity
        offsets_all = self._offset
        if self.appearance_mode == "sh":
            colors_all = self.get_colors()
            shs_view = colors_all.transpose(1, 2).view(-1, 3, 16)
            dir_pp_normalized = -normals_all
            sh2rgb = eval_sh(self.max_sh_degree, shs_view, dir_pp_normalized)
            colors_all = torch.clamp_min(sh2rgb + 0.5, 0.0)
        elif self.appearance_mode == "rgb":
            colors_all = self.get_colors()
        else:
            colors_all = self.get_albedo
        j_indices = torch.arange(len(self.face_t)).cuda()
        j_starts = torch.searchsorted(self.face_indices_t, j_indices, right=False)
        j_ends = torch.searchsorted(self.face_indices_t, j_indices, right=True)
        values, indices = torch.sort(self.faces)
        gs_f_low = torch.searchsorted(values, torch.arange(values[-1] + 1).cuda())
        gs_f_high = torch.searchsorted(
            values, torch.arange(values[-1] + 1).cuda(), right=True
        )
        def project_values(values):
            return gaussian_splat_mesh.gaussian_splat_mesh_cuda(
                vertices,
                self.face_verts.to(torch.int),
                normals_faces,
                self.vertex_neighbors.to(torch.int),
                N_Neighbors,
                indices.to(torch.int),
                gs_f_low.to(torch.int),
                gs_f_high.to(torch.int),
                xyzs_all,
                rotations_all,
                scales_all,
                offsets_all,
                values,
                opacities_all,
                self.face_indices_t.to(torch.int),
                self.barycentrics_t,
                self.pixels_t.to(torch.int),
                j_starts.to(torch.int),
                j_ends.to(torch.int),
                reso,
                1,
            )

        texture_map, normal_map, displacement_map, _ = project_values(colors_all)
        expected_displacement = displacement_map[..., [0]]
        expected_displacement /= texture_map[..., [3]].clamp_min(1e-6)
        median_displacement = displacement_map[..., [1]]
        median = 0.5
        displacement_map = (
            median * median_displacement + (1 - median) * expected_displacement
        )
        normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=-1)
        result = {
            "normal": self.fill_map(normal_map),
            "displacement": self.fill_map(displacement_map),
        }

        if self.appearance_mode != "pbr":
            result["texture"] = self.fill_map(texture_map)
            return result

        def normalize_projected_material(projected):
            alpha = projected[..., [3]]
            material = projected[..., :3] / alpha.clamp_min(1e-6)
            material = torch.where(alpha > 0, material, torch.zeros_like(material))
            return self.fill_map(material)

        result["albedo"] = normalize_projected_material(texture_map)

        roughness_values = self.get_roughness.expand(-1, 3)
        roughness_projected, _, _, _ = project_values(roughness_values)
        result["roughness"] = normalize_projected_material(
            roughness_projected
        )[..., [0]]

        metallic_values = self.get_metallic.expand(-1, 3)
        metallic_projected, _, _, _ = project_values(metallic_values)
        result["metallic"] = normalize_projected_material(
            metallic_projected
        )[..., [0]]

        return result
