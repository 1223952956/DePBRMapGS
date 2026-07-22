import math
import torch
import torch.nn.functional as F


def normalize(x):
    return F.normalize(x, dim=-1, eps=1e-6)


def fresnel_schlick(v_dot_h, f0):
    return f0 + (1.0 - f0) * (1.0 - v_dot_h).pow(5)


def distribution_ggx(n_dot_h, roughness):
    alpha = roughness.square()
    alpha2 = alpha.square()

    denominator = (
        n_dot_h.square() * (alpha2 - 1.0) + 1.0
    ).square()

    return alpha2 / (
        math.pi * denominator + 1e-6
    )


def geometry_schlick_ggx(n_dot_x, roughness):
    k = (roughness + 1.0).square() / 8.0

    return n_dot_x / (
        n_dot_x * (1.0 - k) + k + 1e-6
    )


def shade_pbr(
    positions,
    normals,
    albedo,
    roughness,
    metallic,
    camera_position,
    lighting,
):
    # N × 3
    n = normalize(normals)
    v = normalize(camera_position[None] - positions)

    # K × 3 -> N × K × 3
    l = normalize(lighting["directions"])
    l = l[None].expand(positions.shape[0], -1, -1)

    n_expanded = n[:, None]
    v_expanded = v[:, None]
    h = normalize(v_expanded + l)

    n_dot_l = (
        n_expanded * l
    ).sum(-1, keepdim=True).clamp(0, 1)

    n_dot_v = (
        n_expanded * v_expanded
    ).sum(-1, keepdim=True).clamp(0, 1)

    n_dot_h = (
        n_expanded * h
    ).sum(-1, keepdim=True).clamp(0, 1)

    v_dot_h = (
        v_expanded * h
    ).sum(-1, keepdim=True).clamp(0, 1)

    albedo = albedo[:, None]
    roughness = roughness[:, None]
    metallic = metallic[:, None]

    f0 = (
        0.04 * (1.0 - metallic)
        + albedo * metallic
    )

    D = distribution_ggx(n_dot_h, roughness)
    G = (
        geometry_schlick_ggx(n_dot_v, roughness)
        * geometry_schlick_ggx(n_dot_l, roughness)
    )
    F_term = fresnel_schlick(v_dot_h, f0)

    specular = D * G * F_term / (
        4.0 * n_dot_v * n_dot_l + 1e-6
    )

    kd = (1.0 - F_term) * (1.0 - metallic)
    diffuse = kd * albedo / math.pi

    radiance = lighting["intensity"][None]

    direct = (
        (diffuse + specular)
        * radiance
        * n_dot_l
    ).sum(dim=1)

    ambient = (
        lighting["ambient"][None]
        * gaussian_vals_safe(albedo[:, 0])
        * (1.0 - metallic[:, 0])
    )

    color = direct + ambient

    color = (
        color
        * lighting["exposure"]
        * lighting["white_balance"]
    )

    return color


def gaussian_vals_safe(x):
    return x.clamp(0.0, 1.0)