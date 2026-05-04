from train import train
from config import TrainingConfig

# === GLOBAL SETTINGS ===
TIMESTEPS = 300000
BASELINE_SEEDS = [1, 2, 3, 4]
ALGORITHMS = ["DQN"]

# NOTE:
# In the experiments for the thesis, baseline runs for each algorithm (PPO, A2C, DQN)
# were executed separately. This script represents one example configuration.
# The ALGORITHMS list can be adjusted to run other algorithms accordingly.

def run_baselines():
    print("\n" + "=" * 70)
    print("STARTING DQN BASELINE RUNS")
    print("=" * 70 + "\n")

    for algo in ALGORITHMS:
        for seed in BASELINE_SEEDS:
            config = TrainingConfig()
            config.algorithm = algo
            config.seed = seed
            config.total_timesteps = TIMESTEPS
            config.run_name = f"{algo}_BASE_S{seed}"

            print("\n" + "-" * 60)
            print(f"RUN: {config.run_name}")
            print(f"Algorithm: {config.algorithm}")
            print(f"Seed: {config.seed}")
            print(f"Timesteps: {config.total_timesteps}")
            print("-" * 60)

            train(config)


def main():
    run_baselines()


if __name__ == "__main__":
    main()
