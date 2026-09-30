"""Diagnostic harness: locate the source of NaN observations / solver blow-ups in AER-GO2.

Usage (from the repo root, inside the ``isaac`` conda env)::

    python scripts/diag_aer.py --num_envs 64 --steps 200
    python scripts/diag_aer.py --num_envs 64 --steps 200 --disable add_base_mass
    python scripts/diag_aer.py --num_envs 64 --steps 200 --disable physics_material add_base_mass push_robot

``--disable`` removes event terms from the config before the env is built, which is how
we bisect which randomization is breaking the physics.
"""

import argparse
import sys

# Warp captures ``enable_backward`` when a module is created, which happens at import time,
# so it has to be set before importing anything that defines Warp kernels.
import warp as wp

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import AER.tasks  # noqa: E402, F401  (walks AER.tasks.* -> runs gym.register for AER-GO2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="AER-GO2")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--disable",
        type=str,
        nargs="*",
        default=[],
        help="event term names to disable (e.g. physics_material add_base_mass push_robot)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        choices=["project", "upstream"],
        default="project",
        help="use the project's AER.asserts.unitree.UNITREE_GO2_CFG or isaaclab_assets' one",
    )
    parser.add_argument("--zero", action="store_true", help="apply zero actions instead of random ones")
    parser.add_argument(
        "--newton-actuators",
        type=str,
        choices=["keep", "on", "off"],
        default="keep",
        help="override SimulationCfg.use_newton_actuators",
    )
    parser.add_argument(
        "--reset-joints",
        type=str,
        choices=["none", "scale", "exact"],
        default="none",
        help="inject a joint-reset event (upstream velocity envs always have one)",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli


def _describe(unwrapped, tag: str) -> None:
    """Print a full term-by-term dump of every observation group plus the robot state."""
    robot = unwrapped.scene["robot"]
    om = unwrapped.observation_manager
    print(f"--- {tag} ---")
    for group, names in om._group_obs_term_names.items():
        for name, cfg in zip(names, om._group_obs_term_cfgs[group]):
            try:
                value = cfg.func(unwrapped, **cfg.params)
                finite = bool(torch.isfinite(value).all())
                print(
                    f"  {group}/{name}: shape={tuple(value.shape)} finite={finite} "
                    f"min={float(value.min()):.4g} max={float(value.max()):.4g}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {group}/{name}: ERROR {type(exc).__name__}: {exc}")
    mass = robot.data.body_mass.torch
    inertia = robot.data.body_inertia.torch
    print(
        f"  body_mass: min={float(mass.min()):.6g} max={float(mass.max()):.6g} "
        f"finite={bool(torch.isfinite(mass).all())}"
    )
    print(
        f"  body_inertia: shape={tuple(inertia.shape)} min={float(inertia.min()):.6g} "
        f"max={float(inertia.max()):.6g} finite={bool(torch.isfinite(inertia).all())}"
    )
    names = robot.data.body_names
    print("  joint_names:", list(robot.data.joint_names))
    print("  default_joint_pos[0]:", [round(v, 4) for v in robot.data.default_joint_pos.torch[0].tolist()])
    print("  joint_pos[0]        :", [round(v, 4) for v in robot.data.joint_pos.torch[0].tolist()])
    print("  joint_vel[0]        :", [round(v, 4) for v in robot.data.joint_vel.torch[0].tolist()])
    print(f"  {'body':<16}{'mass':>12}  inertia")
    for body_id, body_name in enumerate(names):
        print(f"  {body_name:<16}{float(mass[0, body_id]):>12.6g}  {inertia[0, body_id].tolist()}")
    for label, tensor in (
        ("root_pos_w", robot.data.root_pos_w.torch),
        ("root_lin_vel_w", robot.data.root_lin_vel_w.torch),
        ("root_ang_vel_w", robot.data.root_ang_vel_w.torch),
        ("joint_pos", robot.data.joint_pos.torch),
        ("joint_vel", robot.data.joint_vel.torch),
        ("applied_torque", robot.data.applied_torque.torch),
    ):
        print(
            f"  {label}: absmax={float(tensor.abs().max()):.6g} "
            f"finite={bool(torch.isfinite(tensor).all())}"
        )


def main() -> None:
    args_cli = _parse_args(sys.argv[1:])

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = args_cli.num_envs

    for name in args_cli.disable:
        if getattr(env_cfg.events, name, None) is not None:
            setattr(env_cfg.events, name, None)
            print(f"[diag] disabled event term: {name}")

    active = [k for k, v in vars(env_cfg.events).items() if v is not None]
    print(f"[diag] task={args_cli.task} num_envs={env_cfg.scene.num_envs} steps={args_cli.steps} events={active}")
    print(f"[diag] physics={type(env_cfg.sim.physics).__name__} dt={env_cfg.sim.dt} decimation={env_cfg.decimation}")

    if args_cli.robot == "upstream":
        from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG as UPSTREAM_GO2_CFG

        env_cfg.scene.robot = UPSTREAM_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print("[diag] robot cfg = isaaclab_assets.UNITREE_GO2_CFG (upstream)")
    else:
        print("[diag] robot cfg = AER.asserts.unitree.UNITREE_GO2_CFG (project)")
    print(f"[diag] robot init_state: pos={env_cfg.scene.robot.init_state.pos} joint_pos={env_cfg.scene.robot.init_state.joint_pos}")

    if args_cli.newton_actuators != "keep":
        env_cfg.sim.use_newton_actuators = args_cli.newton_actuators == "on"
    print(f"[diag] sim.use_newton_actuators = {env_cfg.sim.use_newton_actuators}")

    if args_cli.reset_joints != "none":
        from isaaclab.envs import mdp as lab_mdp
        from isaaclab.managers import EventTermCfg as EventTerm

        if args_cli.reset_joints == "scale":
            position_range, velocity_range = (0.5, 1.5), (0.0, 0.0)  # what upstream velocity envs use
        else:
            position_range, velocity_range = (1.0, 1.0), (0.0, 0.0)  # exactly the default pose
        env_cfg.events.reset_robot_joints = EventTerm(
            func=lab_mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": position_range, "velocity_range": velocity_range},
        )
        print(f"[diag] injected reset_robot_joints position_range={position_range}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        device = unwrapped.device

        env.reset()
        _describe(unwrapped, "after reset")

        for step in range(args_cli.steps):
            if args_cli.zero:
                actions = torch.zeros(env.action_space.shape, device=device)
            else:
                actions = 2 * torch.rand(env.action_space.shape, device=device) - 1
            obs, reward, _terminated, _truncated, _info = env.step(actions)

            obs_nan = {group: bool(torch.isnan(value).any()) for group, value in obs.items()}
            reward_bad = bool(torch.isnan(reward).any())
            if any(obs_nan.values()) or reward_bad:
                print(f"[diag] *** step {step}: obs nan={obs_nan} reward_nan={reward_bad}")
                _describe(unwrapped, f"step {step} (failure)")
                break

            if step % 25 == 0:
                robot = unwrapped.scene["robot"]
                print(
                    f"[diag] step {step}: resets={int(unwrapped.reset_buf.sum())} "
                    f"base_z_min={float(robot.data.root_pos_w.torch[:, 2].min()):.3f} "
                    f"joint_vel_absmax={float(robot.data.joint_vel.torch.abs().max()):.3f}"
                )
        else:
            print(f"[diag] no NaN in {args_cli.steps} steps")

        env.close()


if __name__ == "__main__":
    main()
