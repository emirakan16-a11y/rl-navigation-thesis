from train import train
from config import TrainingConfig

TIMESTEPS = 300000


def run_ppo():
    configs = [
        ("LL", 1e-4, 0.1),
        ("LH", 1e-4, 0.3),
        ("HL", 8e-4, 0.1),
        ("HH", 8e-4, 0.3),
    ]

    for name, lr, clip in configs:
        config = TrainingConfig()
        config.algorithm = "PPO"
        config.seed = 1
        config.total_timesteps = TIMESTEPS
        config.run_name = f"PPO_SENS_{name}"

        config.learning_rate = lr
        config.clip_range = clip

        print("\n" + "-" * 60)
        print(f"RUN: {config.run_name}")
        print(f"Algorithm: {config.algorithm}")
        print(f"learning_rate: {lr}")
        print(f"clip_range: {clip}")
        print("-" * 60)

        train(config)


def run_a2c():
    configs = [
        ("LL", 1e-4, 5),
        ("LH", 1e-4, 50),
        ("HL", 8e-4, 5),
        ("HH", 8e-4, 50),
    ]

    for name, lr, n_steps in configs:
        config = TrainingConfig()
        config.algorithm = "A2C"
        config.seed = 1
        config.total_timesteps = TIMESTEPS
        config.run_name = f"A2C_SENS_{name}"

        config.learning_rate = lr
        config.n_steps = n_steps

        print("\n" + "-" * 60)
        print(f"RUN: {config.run_name}")
        print(f"Algorithm: {config.algorithm}")
        print(f"learning_rate: {lr}")
        print(f"n_steps: {n_steps}")
        print("-" * 60)

        train(config)


def run_dqn():
    configs = [
        ("LL", 0.1, 500),
        ("LH", 0.1, 4000),
        ("HL", 0.5, 500),
        ("HH", 0.5, 4000),
    ]

    for name, exploration_fraction, target_update_interval in configs:
        config = TrainingConfig()
        config.algorithm = "DQN"
        config.seed = 1
        config.total_timesteps = TIMESTEPS
        config.run_name = f"DQN_SENS_{name}"

        config.exploration_fraction = exploration_fraction
        config.target_update_interval = target_update_interval

        print("\n" + "-" * 60)
        print(f"RUN: {config.run_name}")
        print(f"Algorithm: {config.algorithm}")
        print(f"exploration_fraction: {exploration_fraction}")
        print(f"target_update_interval: {target_update_interval}")
        print("-" * 60)

        train(config)


def main():
    run_ppo()
    run_a2c()
    run_dqn()


if __name__ == "__main__":
    main()