# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`PIPER_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`PIPER_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference:https://github.com/agilexrobotics/piper_isaac_sim/tree/master
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Songling/piper_description/piper_description.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        #数组组成：除夹爪外的机械臂关节 + 一个主动夹爪关节 (joint_f1)
        #数组数值：在模拟环境中选取合适的初始姿势
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
            "joint8": 0.0,
        },
    ),
    actuators={
        #机械臂的前四个关节
        "piper_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        #机械臂其余关节
        "piper_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        #夹爪的主动关节
        "piper_hand": ImplicitActuatorCfg(
            joint_names_expr=["joint[7-8]"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


PIPER_HIGH_PD_CFG = PIPER_CFG.copy()
PIPER_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
PIPER_HIGH_PD_CFG.actuators["piper_shoulder"].stiffness = 400.0
PIPER_HIGH_PD_CFG.actuators["piper_shoulder"].damping = 80.0
PIPER_HIGH_PD_CFG.actuators["piper_forearm"].stiffness = 400.0
PIPER_HIGH_PD_CFG.actuators["piper_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
