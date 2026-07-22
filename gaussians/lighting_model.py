import torch
from torch import nn
import torch.nn.functional as F
import math


class LightingModel(nn.Module):
    def __init__(self, num_lights=8, num_views=1):
        super().__init__()

        # 初始化为均匀分布附近的方向
        directions = torch.randn(num_lights, 3, device="cuda")
        directions = F.normalize(directions, dim=-1)

        self._directions = nn.Parameter(directions)

        # 环境光初始化能量
        ambient_init = 0.1

        # 剩余 0.9 平均分配给所有方向光
        per_light_init = (1.0 - ambient_init) / num_lights

        # softplus 的反函数
        intensity_raw_init = math.log(math.expm1(per_light_init))
        ambient_raw_init = math.log(math.expm1(ambient_init))

        self._intensity = nn.Parameter(
            torch.full(
                (num_lights, 3),
                intensity_raw_init,
                device="cuda",
            )
        )

        self._ambient = nn.Parameter(
            torch.full(
                (3,),
                ambient_raw_init,
                device="cuda",
            )
        )

        # 每个训练视角单独校正曝光
        self._log_exposure = nn.Parameter(
            torch.zeros(num_views, 1, device="cuda")
        )

        self._white_balance = nn.Parameter(
            torch.zeros(num_views, 3, device="cuda")
        )

        self.optimizer = None

    def training_setup(self, opt):
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [self._directions],
                    "lr": opt.get(
                        "light_direction_lr",
                        1e-4,
                    ),
                    "name": "direction",
                },
                {
                    "params": [self._intensity],
                    "lr": opt.get("light_lr", 1e-3),
                    "name": "intensity",
                },
                {
                    "params": [self._ambient],
                    "lr": opt.get("light_lr", 1e-3),
                    "name": "ambient",
                },
                {
                    "params": [self._log_exposure],
                    "lr": opt.get("exposure_lr", 1e-3),
                    "name": "exposure",
                },
                {
                    "params": [self._white_balance],
                    "lr": opt.get(
                        "white_balance_lr",
                        5e-4,
                    ),
                    "name": "white_balance",
                },
            ]
        )

    @property
    def directions(self):
        return F.normalize(self._directions, dim=-1)

    @property
    def intensity(self):
        return F.softplus(self._intensity)

    @property
    def ambient(self):
        return F.softplus(self._ambient)

    def exposure(self, view_idx):
        # 限制范围，防止曝光替代材质
        return torch.exp(
            0.5 * torch.tanh(self._log_exposure[view_idx])
        )

    def white_balance(self, view_idx):
        return torch.exp(
            0.2 * torch.tanh(self._white_balance[view_idx])
        )

    def get_light(self, view_idx):
        return {
            "directions": self.directions,
            "intensity": self.intensity,
            "ambient": self.ambient,
            "exposure": self.exposure(view_idx),
            "white_balance": self.white_balance(view_idx),
        }

    def regularization(self):
        direct_energy = self.intensity.sum(dim=0)
        ambient_energy = self.ambient

        channel_energy = direct_energy + ambient_energy

        loss_light_scale = (
            channel_energy - 1.0
        ).square().mean()

        loss_exposure_mean = (
            self._log_exposure.mean()
        ).square()

        loss_exposure_variance = (
            self._log_exposure
            - self._log_exposure.mean()
        ).square().mean()

        loss_white_balance_mean = (
            self._white_balance.mean(dim=0)
        ).square().mean()

        loss_white_balance_variance = (
            self._white_balance
            - self._white_balance.mean(dim=0, keepdim=True)
        ).square().mean()

        return {
            "light_scale": loss_light_scale,
            "exposure_mean": loss_exposure_mean,
            "exposure_variance": loss_exposure_variance,
            "white_balance_mean": loss_white_balance_mean,
            "white_balance_variance": loss_white_balance_variance,
        }
