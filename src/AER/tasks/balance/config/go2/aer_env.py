from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

class AERRewardManager(RewardManager):
    positive_terms = {"track_lin_vel_xy", "track_ang_vel_z", "energy_efficiency"}
    sigma_aux = 0.02

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        positive_reward = torch.zeros_like(self._reward_buf)
        negative_reward = torch.zeros_like(self._reward_buf)
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._episode_sums[name] += value # 用于日志、可视化，不参与奖励计算
            self._step_reward[:, term_idx] = value / dt   # 只用来给人看

            if name in self.positive_terms:
                positive_reward += value
            else:
                negative_reward += value
        self._reward_buf[:] = positive_reward * torch.exp(negative_reward / self.sigma_aux)# 最终返回给训练算法的奖励
        return self._reward_buf

class AERManagerBasedRLEnv(ManagerBasedRLEnv):
    def load_managers(self):
        # call the parent class to load the managers for observations and actions.
        super().load_managers()
        # -- reward manager
        self.reward_manager = AERRewardManager(self.cfg.rewards, self)
