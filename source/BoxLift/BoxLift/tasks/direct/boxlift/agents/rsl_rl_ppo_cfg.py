# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "boxlift"
    logger = "wandb"
    # Asymmetric actor / critic obs. The env returns {"policy": <actor_obs>,
    # "privileged": <extras>}. Map both algorithm sets here: actor consumes "policy"
    # only; critic consumes the concat of "policy" + "privileged" → critic_dim =
    # actor_dim + 136 priv extras. See _get_privileged_obs in boxlift_env.py for the
    # contents of the privileged block (clean obj state, DR samples, reference state,
    # contact forces, VOC kp, etc.).
    obs_groups = {"policy": ["policy"], "critic": ["policy", "privileged"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Bumped from [256, 256] to match boxhinge. Larger network helps with the richer
        # reward (eef_box_rel + per-joint BC) and gives capacity for the VOC curriculum.
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Higher entropy useful during VOC curriculum — keeps the policy exploring
        # non-zero residuals as VOC decays.
        entropy_coef=0.015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )