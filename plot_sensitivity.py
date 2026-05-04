import pandas as pd
import matplotlib.pyplot as plt

SMOOTH_WINDOW = 30
METRIC = "curriculum/terminal_success_rate_all"


def plot_sensitivity(file_name, algo):
    df = pd.read_csv(file_name)

    configs = ["LL", "LH", "HL", "HH"]

    plt.figure(figsize=(10, 6))

    for config in configs:
        col = [
            c for c in df.columns
            if f"{algo}_SENS_{config}" in c
            and METRIC in c
            and "__MIN" not in c
            and "__MAX" not in c
        ][0]

        y = df[col].rolling(SMOOTH_WINDOW, min_periods=1).mean()
        plt.plot(df["Step"], y, label=f"{algo}_{config}")

    plt.xlabel("Training Step")
    plt.ylabel("Terminal Success Rate")
    plt.title(f"Sensitivity Analysis: {algo} Terminal Success Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{algo.lower()}_sensitivity_success_rate.png", dpi=300)
    plt.show()

    print(f"\nFinal Values for {algo}:")
    for config in configs:
        col = [
            c for c in df.columns
            if f"{algo}_SENS_{config}" in c
            and METRIC in c
            and "__MIN" not in c
            and "__MAX" not in c
        ][0]

        final_value = df[col].rolling(SMOOTH_WINDOW, min_periods=1).mean().iloc[-1]
        print(f"{algo}_{config}: {final_value:.4f}")


plot_sensitivity("PPO_SENS.csv", "PPO")
plot_sensitivity("A2C_SENS.csv", "A2C")
plot_sensitivity("DQN_SENS.csv", "DQN")

input("Press Enter to exit...")