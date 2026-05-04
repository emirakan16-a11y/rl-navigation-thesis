import pandas as pd
import matplotlib.pyplot as plt

SMOOTH_WINDOW = 30

def load_curve(file_name, algo):
    df = pd.read_csv(file_name)

    cols = [
        c for c in df.columns
        if f"{algo}_BASE" in c
        and "terminal_success_rate_all" in c
        and "__MIN" not in c
        and "__MAX" not in c
    ]

    df["mean"] = df[cols].mean(axis=1)
    df["std"] = df[cols].std(axis=1)

    df["mean_smooth"] = df["mean"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    df["std_smooth"] = df["std"].rolling(SMOOTH_WINDOW, min_periods=1).mean()

    return df


ppo = load_curve("PPO_BASE.csv", "PPO")
a2c = load_curve("A2C_BASE.csv", "A2C")
dqn = load_curve("DQN_BASE.csv", "DQN")

plt.figure(figsize=(10, 6))

for df, label in [(ppo, "PPO"), (a2c, "A2C"), (dqn, "DQN")]:
    x = df["Step"]
    mean = df["mean_smooth"]
    std = df["std_smooth"]

    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.12)

plt.xlabel("Training Step")
plt.ylabel("Terminal Success Rate")
plt.title("Baseline Comparison: Terminal Success Rate")
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("baseline_terminal_success_rate_smooth.png", dpi=300)
plt.show()

print("Final Values:")
print("PPO:", ppo["mean_smooth"].iloc[-1])
print("A2C:", a2c["mean_smooth"].iloc[-1])
print("DQN:", dqn["mean_smooth"].iloc[-1])

plt.show()

input("Press Enter to exit...")