import torch
from torch.utils.data import Dataset
import numpy as np
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
import json
from utils.common import load_barycentric_coordinates, load_mesh, warp_image
import math
import cv2
import os


def load_dataset(opt):
    if opt.dataset == "avatarhq":
        return ActorHQ(opt)
    elif opt.dataset == "blender":
        return BlenderDataset(opt)


class BaseDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.list = self.get_list(opt)

        self.cameras = self.preload_cameras(opt)
        self.imgs, self.masks, self.normals = self.preload_images(opt)
        self.mesh_path = None

    def __getitem__(self, index):
        return edict(
            img=self.imgs[index],
            mask=self.masks[index],
            camera=self.cameras[index],
            normal=self.normals[index],
        )

    def load_mesh(self, opt):
        self.vertices, self.faces, self.uv_faces, self.uvs = load_mesh(self.mesh_path)
        return load_barycentric_coordinates(
            1024, opt.target_size, self.vertices, self.uvs, self.uv_faces, self.faces
        )

    def __len__(self):
        return len(self.list)

    def get_list(self, opt):
        pass

    def preload_images(self, opt):
        pass

    def preload_cameras(self, opt):
        pass


class ActorHQ(BaseDataset):
    def __init__(self, opt):
        self.pose_idx = opt.pose_idx
        self.data_dir = f"{opt.rootpath}/{opt.scene}/Sequence1/1x"
        super().__init__(opt)
        self.mesh_path = (
            f"{opt.rootpath}/{opt.scene}/Sequence1/objs/{self.pose_idx:08}.obj"
        )

    def get_list(self, opt):
        views_lst = list(range(1, 161))
        return views_lst

    def preload_cameras(self, opt):
        import csv

        cameras = []
        with open(
            self.data_dir + "/calibration.csv", "r", newline="", encoding="utf-8"
        ) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                extr_mat = np.identity(4, np.float32)
                extr_mat[:3, :3] = cv2.Rodrigues(
                    np.array(
                        [float(row["rx"]), float(row["ry"]), float(row["rz"])],
                        np.float32,
                    )
                )[0]
                extr_mat[:3, 3] = np.array(
                    [float(row["tx"]), float(row["ty"]), float(row["tz"])]
                )
                extr_mat = np.linalg.inv(extr_mat)
                extr_mat = torch.from_numpy(extr_mat).cuda().float()

                intr_mat = np.identity(3, np.float32)
                intr_mat[0, 0] = float(row["fx"]) * float(row["w"])
                intr_mat[0, 2] = float(row["px"]) * float(row["w"])
                intr_mat[1, 1] = float(row["fy"]) * float(row["h"])
                intr_mat[1, 2] = float(row["py"]) * float(row["h"])
                intr_mat = torch.from_numpy(intr_mat).cuda().float()
                cameras.append(
                    edict(
                        {
                            "extr": extr_mat,
                            "intr": intr_mat,
                            "width": int(row["w"]),
                            "height": int(row["h"]),
                        }
                    )
                )
        return cameras

    def preload_images(self, opt):
        print("--loading camera images")
        imgs = []
        masks = []
        normals = []
        if opt.vis:
            return (
                [None] * len(self.list),
                [None] * len(self.list),
                [None] * len(self.list),
                [None] * len(self.list),
            )
        for i, view in enumerate(self.list):
            print(f"\r{i}", end="")
            camera_file = f"{self.data_dir}/rgbs/Cam{view:03}/Cam{view:03}_rgb{self.pose_idx:06}.jpg"
            mask_file = (
                f"{self.data_dir}/s_masks/Cam{view:03}_rgb{self.pose_idx:06}.npy"
            )
            normal_file = (
                f"{self.data_dir}/normals/Cam{view:03}_rgb{self.pose_idx:06}.npy"
            )
            image = torch.from_numpy(plt.imread(camera_file)).float().cuda() / 255
            mask = torch.from_numpy(np.load(mask_file)).cuda()
            mask_contour = mask.clone()[..., None]
            if os.path.exists(normal_file):
                normal = (
                    torch.from_numpy(np.load(normal_file)).float().cuda()
                    * torch.tensor([[[1, -1, -1]]]).float().cuda()
                )
                normal[~mask] = 0
                cut_images, cut_mask = warp_image(
                    mask, [image, normal, mask_contour], self.cameras[i]
                )
                normals.append(cut_images[1])
            else:
                cut_images, cut_mask = warp_image(
                    mask, [image, mask_contour], self.cameras[i]
                )
                normals.append(None)
            imgs.append(cut_images[0])
            masks.append(cut_mask)
        return imgs, masks, normals

    def load_mesh(self, opt):
        self.vertices, self.faces, self.uv_faces, self.uvs = load_mesh(self.mesh_path)
        return load_barycentric_coordinates(
            1024, opt.target_size, self.vertices, self.uvs, self.uv_faces, self.faces
        )


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class BlenderDataset(BaseDataset):
    def __init__(self, opt):
        self.path = f"{opt.rootpath}/{opt.scene}/"
        super().__init__(opt)
        self.mesh_path = f"{opt.rootpath}/{opt.scene}/template.obj"

    def get_list(self, opt):
        with open(os.path.join(self.path, "transforms_train.json")) as json_file:
            contents = json.load(json_file)
            self.fovx = contents["camera_angle_x"]
            self.frames = contents["frames"][:80]
        return list(range(len(self.frames)))

    def preload_cameras(self, opt):
        cameras = []
        for idx, frame in enumerate(self.frames):
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3] / opt.pos_scale
            width, height = 1080, 1080
            focalx = fov2focal(self.fovx, width)
            intr = (
                torch.tensor([focalx, 0, width / 2, 0, focalx, height / 2, 0, 0, 1])
                .reshape(3, 3)
                .float()
                .cuda()
            )
            extr = torch.eye(4, dtype=torch.float).cuda()
            extr[:3, :3] = torch.tensor(R).float().cuda()
            extr[:3, 3] = torch.tensor(T).float().cuda()

            cameras.append(
                edict({"extr": extr, "intr": intr, "width": width, "height": height})
            )
        return cameras

    def preload_images(self, opt):
        imgs = []
        masks = []
        normals = []
        for idx, frame in enumerate(self.frames):
            if "png" in frame["file_path"]:
                cam_name = os.path.join(frame["file_path"])
            else:
                cam_name = os.path.join(frame["file_path"] + ".png")
            image_path = os.path.join(self.path, cam_name)
            image = torch.from_numpy(plt.imread(image_path)).float().cuda()
            mask = image[:, :, 3] > 0.2
            image = image[:, :, :3]
            imgs.append(image)
            normals.append(None)
            masks.append(mask)
        return imgs, masks, normals

    def load_mesh(self, opt):
        print("--loading mesh")
        vertices, faces, faces_uvs, uvs = load_mesh(self.mesh_path)
        self.vertices, self.faces, self.uvs, self.uv_faces = (
            vertices,
            faces,
            uvs,
            faces_uvs,
        )
        return load_barycentric_coordinates(
            1024, opt.target_size, self.vertices, self.uvs, self.uv_faces, self.faces
        )
