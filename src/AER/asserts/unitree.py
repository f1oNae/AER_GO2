import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

UNITREE_ROBOT_DIR: str = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree"

UNITREE_GO2_CFG = ArticulationCfg(
    # 定义机器人怎么被加载到仿真里
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        # 是否启用传感器
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    # 定义机器人初始状态
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    # 软限位系数，这里指限制在原始关节范围（硬限位、物理限制）的90%内
    soft_joint_pos_limit_factor=0.9,
    # 执行器。决定控制指令如何作用在机器人关节上（即机器人怎么动起来），分为显式与隐式，前者由用户定义关节力矩，执行器自己算；后者由物理引擎内部处理。
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            # the calf sits behind an extra 1.92:1 knee reduction, so its joint-side torque is
            # higher and its joint-side speed lower than the hip and thigh. Values match ``go2.usd``.
            # 这是控制算法允许电机输出的最大扭矩。相当于软件层面设定的“安全红线”。
            actuator_effort_limit={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43},
            # 这是电机物理上能达到的最大扭矩饱和值。通常为了安全，建议 saturation_effort 等于或略大于 effort_limit。
            saturation_effort={".*_hip_joint": 23.7, ".*_thigh_joint": 23.7, ".*_calf_joint": 45.43},
            # 电机允许旋转的最大角速度。
            actuator_velocity_limit={".*_hip_joint": 30.1, ".*_thigh_joint": 30.1, ".*_calf_joint": 15.70},
            # PD里的P
            stiffness=25.0,
            # PD里的D
            damping=0.5,
            # 关节内部摩擦
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 using DC-Motor actuator model."""