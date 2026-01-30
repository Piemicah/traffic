import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sumo_gym_env import SumoTrafficEnv


def plot_reward_curve(monitor_csv="logs/monitor.csv"):
    """
    Stable-Baselines3 Monitor CSV format:
    First line is a comment starting with '#'
    Then columns: r, l, t (reward, length, time)
    """
    if not os.path.exists(monitor_csv):
        print(f"❌ Monitor file not found: {monitor_csv}")
        return

    df = pd.read_csv(monitor_csv, comment="#")
    rewards = df["r"].values

    # Smooth reward curve (moving average)
    window = min(50, len(rewards))
    if window > 1:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        smooth = rewards

    plt.figure()
    plt.plot(rewards, label="Episode Reward")
    if len(smooth) > 1:
        plt.plot(
            range(window - 1, window - 1 + len(smooth)),
            smooth,
            label=f"Moving Avg ({window})",
        )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plt.legend()


def plot_queue_curve(model_path="ppo_sumo_traffic", sim_steps=1000):
    model = PPO.load(model_path)

    env = SumoTrafficEnv(
        sumo_cfg="config.sumocfg",
        sumo_binary="sumo",
        max_steps=sim_steps,
        min_green=10,
        render_mode="none",
    )

    obs, info = env.reset()

    times = []
    ns_q = []
    ew_q = []
    total_q = []
    rewards = []

    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        times.append(step)
        ns_q.append(info["ns_queue"])
        ew_q.append(info["ew_queue"])
        total_q.append(info["total_queue"])
        rewards.append(reward)

        done = terminated or truncated
        step += 1

    env.close()

    plt.figure()
    plt.plot(times, ns_q, label="NS Queue")
    plt.plot(times, ew_q, label="EW Queue")
    plt.plot(times, total_q, label="Total Queue")
    plt.xlabel("Time Step")
    plt.ylabel("Queue (halted vehicles)")
    plt.title("Queue Curve (Trained PPO Agent)")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(times, rewards, label="Reward per Step")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Reward per Step During Evaluation")
    plt.grid(True)
    plt.legend()


if __name__ == "__main__":
    plot_reward_curve("logs/monitor.csv")
    plot_queue_curve("ppo_sumo_traffic", sim_steps=1000)
    plt.show()
