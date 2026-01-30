from stable_baselines3 import PPO
from sumo_gym_env import SumoTrafficEnv


def main():
    env = SumoTrafficEnv(
        sumo_cfg="config.sumocfg",
        sumo_binary="sumo-gui",
        max_steps=1000,
        min_green=10,
        render_mode="human",
    )

    model = PPO.load("ppo_sumo_traffic")

    obs, info = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()
