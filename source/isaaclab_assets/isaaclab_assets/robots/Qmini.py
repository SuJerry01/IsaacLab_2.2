# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Disney Research robots.

The following configuration parameters are available:

* :obj:`BDX_CFG`: The BD-X robot with implicit Actuator model

Reference:

* https://github.com/rimim/AWD/tree/main/awd/data/assets/go_bdx

"""

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

Qmini_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Disney/BDX/BDX.usd",
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/q1/q1.usd",
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["hip_yaw_.*", "hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*", "ankle_pitch_.*"],
            stiffness={
                "hip_yaw_.*": 100.0,
                "hip_roll_.*": 80.0,
                "hip_pitch_.*": 120.0,
                "knee_pitch_.*": 200.0,
                "ankle_pitch_.*": 200.0,
            },
            damping={
                "hip_yaw_.*": 3.0,
                "hip_roll_.*": 3.0,
                "hip_pitch_.*": 6.0,
                "knee_pitch_.*": 6.0,
                "ankle_pitch_.*": 6.0,
            },
        ),

    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the Disney BD-X robot with implicit actuator model."""
