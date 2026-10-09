# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

from AER.asserts.unitree import UNITREE_GO2_CFG as RobotCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
# from isaaclab.envs import mdp
# import isaaclab_tasks.core.velocity.mdp as mdp
from AER.tasks.balance import mdp

##
# Pre-defined configs
##


##
# Scene definition
##
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        # 原项目 smooth slope 共 10%，内部包含正反坡
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.04),
            platform_width=3.0,
            border_width=0.25,
        ),
        "smooth_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.04),
            platform_width=3.0,
            border_width=0.25,
        ),

        # 近似原项目 rough slope
        # 原项目实际是 slope + random rough 的叠加，内置配置无法直接组合
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.0, 0.05),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        # 原项目楼梯比例：35% / 25%
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.07),
            step_width=0.31,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.07),
            step_width=0.31,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # 对齐原项目 discrete obstacles，而不是密集砖缝 boxes
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_height_mode="choice",
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=(0.05, 0.07),
            num_obstacles=20,
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        # terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        # max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = RobotCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors -- the concrete implementation is selected automatically from the active physics
    # backend (Newton / PhysX / OvPhysX); backend-specific fields such as ``global_world_only`` are
    # documented on the config and ignored by the backends that do not use them.
    height_scanner = RayCasterCfg(
        # 表示扫描仪相对于基座向上偏移 2 米。
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        # 表示所有射线都对齐于机器人的偏航角
        ray_alignment="yaw",
        # 扫描点间距和扫描范围
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        # 是否查看射线轨迹
        debug_vis=False,
        # 这表示该扫描仪不检测机器人自身、墙壁或其他障碍物，只关心“脚底下的地面在哪儿”。
        mesh_prim_paths=["/World/ground"],
        # 只检测全局世界坐标系中的物体，不检测其他子环境或动态物体。这确保了在多环境并行训练时，每个机器人只扫描自己环境里的地面
        global_world_only=True,
    )
    # 足底接触力传感器.它会对机器人的每一个身体部件（特别是四个脚掌）都创建一个接触力监测点。
    # 保留最近 3 步的接触力历史数据。额外记录每条腿离地（悬空）的持续时间（即"腾空时间"）。
    # 判断步态相位、计算奖励
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True)
    # lights.这纯粹是视觉美化，不会影响物理仿真或训练效果。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class ComplexSceneCfg(RobotSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=0,
        use_terrain_origins = True,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        # rel_heading_envs=0.25,
        heading_command=False,
        # heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.2), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 6.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
        },
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # 机体根部线速度（坐标系：asset/root frame，也就是机器人根坐标系，不是世界坐标系）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0), scale=1.0)
        # 机体根部角速度（坐标系：asset/root frame）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        # 重力方向在机器人根坐标系下的投影（坐标系：asset/root frame）
        # 常用来表征机器人当前姿态，比如有没有侧倾、俯仰
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        # 速度指令（由 command manager 生成）
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        # 相对关节位置（相对默认关节位置的偏差）
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        # 相对关节速度（相对默认关节速度的偏差）
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        # 关节输出力 / 力矩（单位通常是 N 或 N·m）
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        # 上一时刻的动作输出
        # 一般就是上一拍策略网络送给环境的 action，可用于提升控制平滑性或提供时序信息
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # 高度扫描结果（坐标系：sensor frame，也就是 height_scanner 这个传感器自己的坐标系）
        # 本质上是传感器射线打到地面后得到的局部高度信息
        # 这里的 sensor_cfg 指向你在 scene 里定义好的 "height_scanner"
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.history_length = 30
    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class StudentCfg(ObsGroup):
        """Deployable observations used by the recurrent student policy."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    student: StudentCfg = StudentCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0),
                               scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100),
                                    params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        height_scanner = ObsTerm(func=mdp.height_scan,
                                 params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                                 clip=(-1.0, 5.0),
                                 )

    critic: CriticCfg = CriticCfg()


# 域随机化内容大都放在这里面
@configclass
class EventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    foot_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range": (0.6, 1.2),
                "dynamic_friction_range": (0.6, 1.2),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
            },
        )

    base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-1.0, 1.0),
                "operation": "add",
                "recompute_inertia": True,
            },
        )

    base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.015, 0.015),
                    "z": (-0.01, 0.01),
                },
            },
        )

    actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )
    reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (-0.5, 0.5),
            },
        )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.001)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # 惩罚关节输出力矩过大
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    # 惩罚关节速度
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    # 惩罚关节加速度
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_2 = RewTerm(func=mdp.action_smoothness_2, weight=-0.01)
    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.02,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
        },
    )

    energy_efficiency = RewTerm(
        func=mdp.energy_efficiency,  # 指向你的函数
        weight=3,  # 设置权重，正数表示奖励
        params={
            # 这里传入函数签名中的参数
            "sigma_lin": 1000.0,
            "sigma_ang": 500.0,
            "clip_lin": 0.02,
            "clip_ang": 0.02,
            # asset_cfg 通常有默认值 SceneEntityCfg("robot")，可以不传
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-30.0,
        params={
            "target_height": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 机身撞地
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # 机器人倾斜太厉害，超过阈值0.8
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": 0.8}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    lin_vel_cmd_levels = CurriculumTermCfg(func=mdp.lin_vel_cmd_levels)

@configclass
class ComplexCurriculumCfg:
    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)

##
# Environment configuration
##


@configclass
class Go2DemoEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Simulation settings — shared physics preset (PhysX + MJWarp) for all rough-terrain envs
    sim: SimulationCfg = SimulationCfg(physics=RoughPhysicsCfg())
    sim.use_newton_actuators = True
    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def play_mode(self):
        """Play-mode overrides shared by the velocity-tracking environments."""
        super().play_mode()
        # spawn the robot r
        #
        # andomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # remove random pushing events
        self.events.base_external_force_torque = None
        self.events.push_robot = None


class ComplexEnvCfg(Go2DemoEnvCfg):
    scene: RobotSceneCfg = ComplexSceneCfg(num_envs=4096, env_spacing=2.5)
    curriculum: CurriculumCfg = ComplexCurriculumCfg()