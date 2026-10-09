# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from collections.abc import Sequence
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def energy_efficiency(env: ManagerBasedRLEnv, asset_cfg=SceneEntityCfg("robot"), sigma_lin=1000.0, sigma_ang=500.0, clip_lin=0.2, clip_ang=0.2) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # 关节速度
    joint_vel = asset.data.joint_vel.torch
    # 力矩
    joint_torque = asset.data.applied_torque.torch
    # 计算分子
    energy = torch.sum(torch.abs(joint_torque * joint_vel), dim=1)

    base_lin_vel_x = asset.data.root_lin_vel_b.torch[:, 0]
    base_ang_vel_z = asset.data.root_ang_vel_b.torch[:, 2]

    # 计算分母
    denom = (
        sigma_lin * torch.clamp(torch.abs(base_lin_vel_x), min=clip_lin)
        + sigma_ang * torch.clamp(torch.abs(base_ang_vel_z), min=clip_ang)
    )

    return torch.exp(-energy / denom)


def feet_slip(env: ManagerBasedRLEnv, sensor_cfg, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(contacts * torch.sum(torch.square(feet_vel_xy), dim=-1), dim=1)

class action_smoothness_2(ManagerTermBase):
    """Penalize the second-order finite difference of policy actions."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros_like(env.action_manager.action)
        self._history_length = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_prev_action[env_ids] = 0.0
        self._history_length[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        second_difference = (
            env.action_manager.action
            - 2.0 * env.action_manager.prev_action
            + self._prev_prev_action
        )
        penalty = torch.sum(torch.square(second_difference), dim=1)
        valid_history = self._history_length >= 2

        self._prev_prev_action.copy_(env.action_manager.prev_action)
        self._history_length.add_(1).clamp_(max=2)

        return penalty * valid_history.float()