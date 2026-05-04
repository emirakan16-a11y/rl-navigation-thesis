"""
Main training script with curriculum learning and algorithm swapping.
"""

import os
import gc
import wandb
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from environments import (
    RobustNoRenderLayout1a, RobustNoRenderLayout1b,
    RobustNoRenderLayout2a, RobustNoRenderLayout2b,
    RobustNoRenderLayout3a, RobustNoRenderLayout3b,
    make_curriculum_env
)
from config import TrainingConfig
from curriculum_manager import CurriculumManager


def setup_directories(run_id, base_path=None):
    """Create directories on external SSD"""
    if base_path is None:
        base_path = "."

    if not os.path.exists(base_path):
        try:
            os.makedirs(base_path, exist_ok=True)
        except Exception:
            print(f"Warning: Cannot create {base_path}, using current directory")
            base_path = "."

    dirs = {
        "checkpoint": f"{base_path}/checkpoints/{run_id}",
        "episode": f"{base_path}/episode_logs/{run_id}",
        "curriculum": f"{base_path}/curriculum_logs/{run_id}",
        "script": f"{base_path}/script_logs/{run_id}",
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    print(f"✓ Saving logs to: {base_path}")
    return dirs


def create_environments(config, layout_classes):
    """Create vectorized environments"""
    base_seed = config.seed
    env_seeds = [base_seed + i * 1000 for i in range(config.num_envs)]

    terminal_interval = config.terminal_change_interval if config.enable_terminal_randomization else 0
    layout_interval = config.layout_change_interval if config.enable_layout_switching else 0

    initial_reward_scales = config.get_initial_reward_scales()

    envs = DummyVecEnv([
        make_curriculum_env(
            layout_classes=layout_classes,
            max_episode_steps=config.max_episode_steps,
            steps_until_hallway=config.steps_until_hallway,
            reward_scales=initial_reward_scales,
            terminal_change_interval=terminal_interval,
            layout_change_interval=layout_interval,
            episodes_per_layout=config.episodes_per_layout,
            seed=env_seeds[i]
        ) for i in range(config.num_envs)
    ])

    return envs


def create_model(config, envs):
    """Create RL model based on algorithm selection"""
    hyperparams = config.get_hyperparams()
    algo_kwargs = config.algorithm_kwargs.get(config.algorithm, {})

    model_kwargs = {
        **algo_kwargs,
        **hyperparams,
        "verbose": 1,
        "tensorboard_log": f"runs/{config.run_name}_{wandb.run.id}",
        "seed": config.seed,
        "device": "auto"
    }

    policy_type = algo_kwargs["policy_type"]
    model_kwargs = {k: v for k, v in model_kwargs.items() if k != "policy_type"}

    if config.algorithm == "PPO":
        allowed_keys = {
            "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef",
            "vf_coef", "max_grad_norm", "tensorboard_log",
            "verbose", "seed", "device"
        }
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed_keys}
        model = PPO(policy_type, envs, **filtered_kwargs)

    elif config.algorithm == "A2C":
        allowed_keys = {
            "learning_rate", "n_steps", "gamma", "gae_lambda",
            "ent_coef", "vf_coef", "max_grad_norm", "rms_prop_eps",
            "use_rms_prop", "normalize_advantage", "tensorboard_log",
            "verbose", "seed", "device"
        }
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed_keys}
        model = A2C(policy_type, envs, **filtered_kwargs)

    elif config.algorithm == "DQN":
        allowed_keys = {
            "learning_rate", "buffer_size", "learning_starts",
            "batch_size", "tau", "gamma", "train_freq",
            "gradient_steps", "target_update_interval",
            "exploration_fraction", "exploration_initial_eps",
            "exploration_final_eps", "max_grad_norm",
            "tensorboard_log", "verbose", "seed", "device"
        }
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed_keys}
        model = DQN(policy_type, envs, **filtered_kwargs)

    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    return model


class CurriculumCallback(WandbCallback):
    """Callback that manages curriculum transitions"""

    def __init__(self, curriculum_manager, model, envs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_manager = curriculum_manager
        self.model_ref = model
        self.envs = envs
        self.previous_layout = None
        self.layout_change_count = 0

    def _on_step(self) -> bool:
        """Check for curriculum transitions after each step"""
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    hallway_reached = info.get("reached_hallway", False) or info.get("hallway_reached", False)
                    terminal_reached = info.get("reached_terminal", False)

                    current_layout = None
                    if "curriculum" in info and "current_layout" in info["curriculum"]:
                        current_layout = info["curriculum"]["current_layout"]

                    if current_layout is not None and current_layout != self.previous_layout:
                        if self.previous_layout is not None:
                            self.layout_change_count += 1
                            print(f"\n📍 LAYOUT CHANGE: {self.previous_layout} → {current_layout}")
                            print(f"   Total layout changes: {self.layout_change_count}")
                        self.previous_layout = current_layout

                    self.curriculum_manager.log_episode(
                        hallway_reached,
                        terminal_reached,
                        current_layout
                    )

                    status = self.curriculum_manager.get_status()
                    log_dict = {
                        "curriculum/hallway_success_rate_50": status["hallway_success_rate_50"],
                        "curriculum/terminal_success_rate_50": status["terminal_success_rate_50"],
                        "curriculum/hallway_success_rate_all": status["hallway_success_rate_all"],
                        "curriculum/terminal_success_rate_all": status["terminal_success_rate_all"],
                        "curriculum/current_layout": self._layout_to_index(current_layout),
                        "curriculum/layout_changes": self.layout_change_count,
                    }

                    for key, value in status.items():
                        if key.startswith("family_"):
                            log_dict[f"curriculum/{key}"] = value

                    wandb.log(log_dict, step=self.num_timesteps)

        return super()._on_step()

    def _layout_to_index(self, layout_name):
        """Convert layout name to numeric index for visualization"""
        if layout_name is None:
            return -1

        layout_map = {
            "RobustNoRenderLayout1a": 0,
            "RobustNoRenderLayout1b": 1,
            "RobustNoRenderLayout2a": 2,
            "RobustNoRenderLayout2b": 3,
            "RobustNoRenderLayout3a": 4,
            "RobustNoRenderLayout3b": 5,
        }
        return layout_map.get(layout_name, -1)

    def _update_hyperparameters(self, new_hyperparams):
        """Update model hyperparameters dynamically"""
        for param, value in new_hyperparams.items():
            if hasattr(self.model_ref, param):
                setattr(self.model_ref, param, value)
                print(f"  Updated {param} = {value}")

    def _update_reward_scales(self, new_reward_scales):
        """Update environment reward scales"""
        for env_idx in range(self.envs.num_envs):
            env = self.envs.envs[env_idx].unwrapped
            while hasattr(env, "env"):
                env = env.env

            if hasattr(env, "reward_scales"):
                env.reward_scales.update(new_reward_scales)
                print(f"  Updated reward scales for env {env_idx}")


def setup_callbacks(config, dirs, curriculum_manager, model, envs):
    """Set up training callbacks"""
    callbacks = [
        CurriculumCallback(
            curriculum_manager=curriculum_manager,
            model=model,
            envs=envs,
            model_save_path=f"models/{wandb.run.id}_{config.algorithm}_{config.run_name}",
            verbose=2,
        ),
    ]
    return callbacks


def train(config):
    """Main training function"""
    config.print_config()

    with wandb.init(
        project=config.project_name,
        name=config.run_name,
        config=config.to_dict(),
        sync_tensorboard=True
    ) as run:

        wandb.save(__file__)
        wandb.save("config/training_config.py")

        dirs = setup_directories(wandb.run.id, base_path=config.log_base_path)

        layout_classes = [
            RobustNoRenderLayout1a,
        ]

        print(f"\nUsing {len(layout_classes)} layouts")

        curriculum_manager = CurriculumManager(
            hyperparam_schedule=None,
            reward_schedule=None,
            episodes_per_layout=config.episodes_per_layout,
            verbose=True
        )

        print("Creating environments...")
        envs = create_environments(config, layout_classes)

        print(f"Creating {config.algorithm} model...")
        model = create_model(config, envs)

        callbacks = setup_callbacks(config, dirs, curriculum_manager, model, envs)

        try:
            print(f"\nStarting training...")
            print(f"Total timesteps: {config.total_timesteps:,}\n")

            model.learn(
                total_timesteps=config.total_timesteps,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name=f"{config.algorithm}_{config.run_name}"
            )

            final_model_path = f"{dirs['checkpoint']}/final_model.zip"
            model.save(final_model_path)
            print(f"\nTraining completed! Final model saved to: {final_model_path}")

        finally:
            envs.close()
            del model
            gc.collect()


def main():
    """Main entry point"""
    import sys
    import multiprocessing

    try:
        if sys.platform.startswith("win"):
            multiprocessing.set_start_method("spawn", force=True)
        else:
            multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    config = TrainingConfig()

    # manuel tekli test için istersen burada bırakabilirsin
    config.seed = 1
    config.total_timesteps = 300000
    config.run_name = "PPO_BASE_S1"

    train(config)


if __name__ == "__main__":
    main()