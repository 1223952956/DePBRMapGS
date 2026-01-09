import argparse
import torch
import random
import numpy as np
from options.options import add_arguments, load_yaml_options
from dataloader import load_dataset

from tqdm import tqdm
from utils.loss import l1, normal_error
from utils.ssim import ssim
from utils.common import log_loss, save_mesh
from utils.mesh_renderer import render_white_faces_with_black_edges
import os
import matplotlib.pyplot as plt
from lpips import LPIPS
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict

rendergs = None

loss_weights = {
    "photometric": 1,
    "ssim": 0.1,
    "surf_normal": 0.1,
    "gt_normal": 1,
    "distortion": 0.01,
    "reg": 50,
}


def render(gaussian_vals, calib, bg):
    global rendergs

    render_ret = rendergs(
        gaussian_vals, bg, calib["extr"], calib["intr"], calib["width"], calib["height"]
    )
    return render_ret


def color_iter(opt, gaussian_model, data, vgg, iter):
    gaussian_model.optimizer.zero_grad()
    gaussian_model.optimizer_g.zero_grad()
    gaussian_vals = gaussian_model.get_gaussian_vals()
    bg = torch.rand(3).float().cuda()
    gt_img = data.img.clone()
    gt_img[~data.mask] = bg
    render_ret = render(gaussian_vals, data.camera, bg)
    viewspace_point_tensor = render_ret["viewspace_points"]
    visibility_filter = render_ret["visibility_filter"]
    radii = render_ret["radii"]
    losses = {
        "photometric": l1(gt_img, render_ret["render"].permute(1, 2, 0)),
        "ssim": 1 - ssim(gt_img, render_ret["render"].permute(1, 2, 0)),
        "reg": (gaussian_model.L @ gaussian_vals["verts"]).square().mean()
        if iter < opt.deform_iter
        else torch.tensor(0).cuda().float(),
    }
    if not gaussian_model.threed:
        losses.update(
            {
                "surf_normal": normal_error(
                    render_ret["rend_normal"], render_ret["surf_normal"]
                )
                if iter > opt.total_iter - opt.last_iter
                else torch.tensor(0).cuda().float(),
                "gt_normal": normal_error(
                    data.normal.permute(2, 0, 1), render_ret["rend_normal"]
                )
                if data.normal is not None
                else torch.tensor(0).cuda(),
                "distortion": render_ret["rend_dist"].mean()
                if iter > opt.total_iter - opt.last_iter
                else torch.tensor(0).cuda().float(),
            }
        )
    global loss_weights
    all_loss = sum([loss_weights[k] * losses[k] for k in losses])
    all_loss.backward()
    gaussian_model.optimizer_g.step()
    gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)
    if iter > opt.global_iter:
        gaussian_model.optimizer.step()
        gaussian_model.update_learning_rate(iter)
    if iter > 2000:
        gaussian_model.update_global_rate()
    if iter < opt.vertice_iter:
        if iter % 50 == 0:
            ### vertex realignment
            gaussian_model.revise_vertices()
            with torch.no_grad():
                ### reset displacement
                face_offset_avg, face_offset_abs, face_offset_count = (
                    gaussian_model.averate_offset()
                )
                min_count = face_offset_count.min()
                gaussian_model.attach_splat(face_offset_count == 0)
                if (iter < opt.d_iter) and (
                    face_offset_avg.max() > 0.03 / opt.d_iter * iter
                ):
                    gaussian_model.reset_offset()
                writer.add_scalar("displacement/max", face_offset_abs.max(), iter)
                writer.add_scalar("displacement/count", min_count, iter)
    ### walk on triangle
    gaussian_model.update_triangle()
    ### densification and pruning
    if (
        (iter > opt.deform_iter)
        and (iter < opt.total_iter - 5000)
        and ((iter % 200) == 0)
    ):
        size_threshold = 20 if iter > 3000 else None
        gaussian_model.densify_and_prune(0.0001, 0.005, 5, size_threshold, radii)
    if (
        (iter > opt.deform_iter)
        and ((iter + 1) % 2000 == 0)
        and (iter < opt.total_iter - 10000)
    ):
        gaussian_model.reset_opacity()
    return losses


def train(opt, gaussian_model, train_data, writer, map_mask=None):
    global rendergs
    if opt.mode == "no-3d":
        print("not using 3dgs")
        opt.no_3d = True
    output_dir = f"{opt.group}/{opt.exp}"
    os.makedirs(output_dir, exist_ok=True)
    gaussian_model.training_setup()
    all_iteration = opt.total_iter
    progress = tqdm(range(all_iteration))
    stack = []
    vgg = LPIPS(net="vgg").cuda()
    opt.img_save_dir = f"{output_dir}/images/"
    os.makedirs(opt.img_save_dir, exist_ok=True)
    for iter in progress:
        if (iter == 0) or ((iter + 1) % 5000 == 0):
            evaluate(opt, gaussian_model, train_data, writer, iter)
            save_properties(opt, output_dir, gaussian_model, train_data, writer, iter)
        if not opt.no_3d:
            if iter == opt.total_iter - opt.last_iter:
                gaussian_model.threed = False
                from gaussians.render_2dgs import render3

                rendergs = render3
                print("switch to 2dgs")
            elif iter == opt.d_iter:
                gaussian_model.threed = True
                from gaussians.render_3dgs import render3

                rendergs = render3
                print("switch to 3dgs")
        if (opt.sh) and (iter % 10000 == 0):
            gaussian_model.oneupSHdegree()
        if stack == []:
            stack = torch.randperm(len(train_data)).squeeze().tolist()
        idx = stack.pop()
        data = train_data[idx]
        losses = color_iter(opt, gaussian_model, data, vgg, iter)
        progress.set_postfix(
            {key: f"{item.tolist():.03f}" for key, item in losses.items()}
        )

        if ((iter + 1) % 50) == 0:
            log_loss(writer, losses, iter)

        if iter < opt.deform_iter:
            if (iter + 1) % 50 == 0:
                gaussian_model.cal_scale()
        else:
            gaussian_model.scales.requires_grad = True
    save_properties(opt, output_dir, gaussian_model, train_data, writer, iter)
    gaussian_model.capture(f"{output_dir}/model.ckpt")


@torch.no_grad()
def save_properties(opt, output_dir, gaussian_model, train_data, writer, iter):
    for idx in opt.vis_idx:
        data = train_data[idx]
        if opt.dataset == "avatarhq":
            camera = edict(
                {
                    "extr": torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [-0.0, -1.0, -0.0, 1.0],
                            [-0.0, -0.0, -1.0, 3.0],
                            [-0.0, -0.0, -0.0, 1.0],
                        ]
                    )
                    .float()
                    .cuda(),
                    "intr": data.camera["intr"],
                    "width": data.camera["width"],
                    "height": data.camera["height"],
                }
            )
        else:
            camera = data.camera
        edge_img = render_white_faces_with_black_edges(
            gaussian_model.get_vertices[None],
            train_data.faces[None],
            camera,
        )[0]
        edge_img[edge_img[..., 3] == 0] = 1
        count = iter + 1
        plt.imsave(
            os.path.join(opt.img_save_dir, f"edge-{count}.png"),
            edge_img.clamp(0, 1).cpu().numpy(),
        )
    if iter > opt.total_iter * 0.9:
        texture_map, normal_map, displacement_map = (
            gaussian_model.project_property_to_uv(opt)
        )
        plt.imsave(f"{output_dir}/texture.png", texture_map.clamp(0, 1).cpu().numpy())
        plt.imsave(f"{output_dir}/normal.png", (normal_map / 2 + 0.5).cpu().numpy())
        np.save(
            f"{output_dir}/displacement.npy", displacement_map.squeeze().cpu().numpy()
        )
        save_mesh(
            f"{output_dir}/mesh.obj",
            gaussian_model.get_vertices,
            train_data.faces,
            train_data.uv_faces,
            train_data.uvs,
        )


@torch.no_grad()
def evaluate(opt, gaussian_model, train_data, writer, iter):
    gaussian_vals = gaussian_model.get_gaussian_vals()
    bg = torch.ones(3).float().cuda()
    for idx in opt.vis_idx:
        data = train_data[idx]
        data.img[~data.mask] = 1
        count = iter + 1
        render_ret = render(gaussian_vals, data.camera, bg)
        writer.add_image(f"render/rgb-{idx}", render_ret["render"], iter)
        writer.add_image(f"render/gt-{idx}", data.img.permute(2, 0, 1), iter)
        if not gaussian_model.threed:
            h, w = render_ret["rend_normal"].shape[1:]
            render_ret["rend_normal"] = (
                data.camera["extr"][:3, :3] @ render_ret["rend_normal"].reshape(3, -1)
            ).reshape(3, h, -1)
            render_ret["rend_normal"] = (
                render_ret["rend_alpha"] * render_ret["rend_normal"]
                + 1
                - render_ret["rend_alpha"]
            )
            render_ret["surf_normal"] = (
                render_ret["rend_alpha"] * render_ret["surf_normal"]
                + 1
                - render_ret["rend_alpha"]
            )
            writer.add_image(f"render/normal-{idx}", render_ret["rend_normal"], iter)
            writer.add_image(
                f"render/depth-{idx}",
                render_ret["surf_depth"] / render_ret["surf_depth"].max(),
                iter,
            )
            if data.normal is not None:
                writer.add_image(
                    f"render/ref-normal-{idx}", data.normal.permute(2, 0, 1), iter
                )
            else:
                writer.add_image(
                    f"render/surf-normal-{idx}", render_ret["surf_normal"], iter
                )
        plt.imsave(
            f"{opt.img_save_dir}/render-{count}.png",
            render_ret["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy(),
        )


@torch.no_grad()
def render_frame(opt, gaussian_model, train_data, fname):
    global rendergs
    from gaussians.render_2dgs import render3

    rendergs = render3
    opt.threed = False
    gaussian_model.threed = False

    data = train_data[opt.vis_idx[0]]
    bg = torch.ones(3).float().cuda()
    gaussian_vals = gaussian_model.get_gaussian_vals()
    render_ret = render(gaussian_vals, data.camera, bg)
    img = render_ret["render"].permute(1, 2, 0)
    plt.imsave(fname + "render.png", img.clamp(0, 1).cpu().numpy())
    edge_img = render_white_faces_with_black_edges(
        gaussian_model.get_vertices[None], train_data.faces[None], data.camera, None
    )
    edge_img[edge_img[..., 3] == 0] = 1
    alpha = render_ret["rend_alpha"][0]
    edge_img = edge_img[0]
    res = edge_img[..., :3] * (1 - alpha[..., None]) + img * alpha[..., None]
    plt.imsave(fname + "render.png", res.clamp(0, 1).cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    opt_arg = parser.parse_args()
    opt = load_yaml_options(opt_arg)
    random.seed(27519)
    np.random.seed(27519)
    torch.manual_seed(27519)
    torch.set_default_device(f"cuda:{opt.cuda}")

    train_data = load_dataset(opt)
    mesh = train_data.load_mesh(opt)

    from gaussians.render_2dgs import render3

    rendergs = render3
    if not opt.vis:
        writer = SummaryWriter(f"{opt.group}/{opt.exp}")
        from gaussians.gaussian_model import GaussianModelTrain

        gaussian_model = GaussianModelTrain(opt, "not loading", mesh)
        train(opt, gaussian_model, train_data, writer)
    else:
        if opt.app == "vis":
            from gaussians.gaussian_model import GaussianModelVis

            gaussian_model = GaussianModelVis(
                opt, f"{opt.group}/{opt.exp}/model.ckpt", mesh, False
            )
        elif opt.app == "editcolor":
            from gaussians.gaussian_model import GaussianModelEditColor

            gaussian_model = GaussianModelEditColor(
                opt, f"{opt.group}/{opt.exp}/model.ckpt", mesh, False
            )
        elif opt.app == "editgeometry":
            from gaussians.gaussian_model import GaussianModelEditGeometry

            gaussian_model = GaussianModelEditGeometry(
                opt, f"{opt.group}/{opt.exp}/model.ckpt", mesh, False
            )
        render_frame(
            opt, gaussian_model, train_data, f"{opt.group}/{opt.exp}/{opt.app}-"
        )
