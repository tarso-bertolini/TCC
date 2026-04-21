import os
import torch
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from data_generator import generate_microgrid_data
from microgrid_env import MicrogridEnv

def get_device() -> str:
    """Returns 'mps' if Apple Silicon GPU is available, otherwise 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def evaluate_model(model, env, num_steps: int) -> float:
    """Evaluates a trained model and returns the total reward."""
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Generate Data
    print("Generating synthetic telemetry data...")
    df = generate_microgrid_data(days=365)
    
    # Split: 300 days train, 65 days test
    train_size = 300 * 24
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    # 2. Setup Environments
    train_env = MicrogridEnv(df_train)
    test_env = MicrogridEnv(df_test)
    
    # Validate environment
    check_env(train_env)
    
    # 3. Initialize Agents
    print("\nInitializing PPO...")
    ppo_model = PPO("MlpPolicy", train_env, device=device, verbose=1)
    
    print("\nInitializing DQN...")
    dqn_model = DQN("MlpPolicy", train_env, device=device, verbose=1)
    
    # 4. Train Agents
    training_steps = 50000
    print(f"\nTraining PPO for {training_steps} steps...")
    ppo_model.learn(total_timesteps=training_steps)
    
    print(f"\nTraining DQN for {training_steps} steps...")
    dqn_model.learn(total_timesteps=training_steps)
    
    # 5. Benchmarking
    print("\n--- Evaluation on Test Set ---")
    
    # Baseline: Do nothing (Always Idle)
    print("Evaluating Baseline (Always Idle)...")
    obs, _ = test_env.reset()
    baseline_reward = 0.0
    for _ in range(len(df_test) - 1):
        obs, reward, terminated, truncated, info = test_env.step(1) # 1 is Idle
        baseline_reward += reward
        if terminated: break
        
    print("Evaluating PPO...")
    ppo_reward = evaluate_model(ppo_model, test_env, len(df_test) - 1)
    
    print("Evaluating DQN...")
    dqn_reward = evaluate_model(dqn_model, test_env, len(df_test) - 1)
    
    print("\n--- Benchmarking Results (Total Reward / Negative Cost) ---")
    print(f"Baseline (Idle): {baseline_reward:.2f}")
    print(f"PPO Agent:       {ppo_reward:.2f}")
    print(f"DQN Agent:       {dqn_reward:.2f}")

    if ppo_reward > baseline_reward:
        print("PPO successfully reduced energy costs compared to baseline!")
    if dqn_reward > baseline_reward:
        print("DQN successfully reduced energy costs compared to baseline!")

if __name__ == "__main__":
    main()
