import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from sumo_gym_env import SumoTrafficEnv


def main():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        env = SumoTrafficEnv(
            sumo_cfg="config.sumocfg",
            sumo_binary="sumo-gui",
            max_steps=1000,
            min_green=10,
            render_mode="none",
        )
        # Monitor records episode rewards + lengths
        return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    env = DummyVecEnv([make_env])
    env = VecMonitor(env, filename=os.path.join(log_dir, "vecmonitor"))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=200_000)
    model.save("ppo_sumo_traffic")

    print("✅ Training complete. Model saved as ppo_sumo_traffic.zip")
    print(f"📁 Logs saved inside: {log_dir}/")


if __name__ == "__main__":
    main()
