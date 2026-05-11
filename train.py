"""
train.py
========
Main training entry point for the Smart Classroom RL project.

Usage:
    python train.py --config config/qlearning_v1.yaml
    python train.py --config config/qlearning_v2.yaml

What this script does:
  1. Loads hyperparameters from the specified YAML config.
  2. Initialises the ClassroomEnvironment and QLearningAgent.
  3. Runs training for N episodes.
  4. Saves the Q-table model, episode CSV, and updates log.json.
  5. Generates and saves training-curve plots.
"""

import argparse
import os
import sys

# ---- Make sure project root is on path ----
sys.path.insert(0, os.path.dirname(__file__))

from environment import ClassroomEnvironment
from agent import QLearningAgent
from utils import (
    load_config,
    generate_run_id,
    log_experiment,
    save_results_csv,
    plot_training_curves,
    print_header,
    print_episode_summary,
    compute_episode_metrics,
)


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Q-Learning agent for Smart Classroom Energy Saver"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/qlearning_v1.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Override output CSV path (default: taken from config)",
    )
    return parser.parse_args()


# ===========================================================================
# Training loop
# ===========================================================================

def train(config, csv_out_override=None):
    """
    Full training loop.

    Args:
        config          (dict): Parsed YAML config.
        csv_out_override(str) : Optional override for CSV save path.

    Returns:
        dict: Summary metrics from the final training run.
    """
    print_header("Smart Classroom Energy Saver — Q-Learning Training")

    # ---- Unpack config ----
    episodes      = config["episodes"]
    max_steps     = config["max_steps"]
    lr            = config["learning_rate"]
    gamma         = config["gamma"]
    epsilon_start = config["epsilon_start"]
    epsilon_min   = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]
    seed          = config.get("seed", 42)
    model_path    = config.get("model_path", "models/q_table.pkl")
    log_path      = config.get("log_path", "results/log.json")
    csv_path      = csv_out_override or config.get("csv_path", "results/results_1.csv")
    plot_path     = config.get("plot_path", "results/training_curves.png")
    notes         = config.get("notes", "")
    experiment_name = config.get("experiment_name", "Q-Learning Experiment")

    print(f"\n  Experiment : {experiment_name}")
    print(f"  Episodes   : {episodes}")
    print(f"  Steps/ep   : {max_steps}")
    print(f"  LR / γ / ε : {lr} / {gamma} / {epsilon_start}")
    print(f"  Seed       : {seed}\n")

    # ---- Initialise environment + agent ----
    env   = ClassroomEnvironment(max_steps=max_steps, seed=seed)
    agent = QLearningAgent(
        state_size    = env.get_state_size(),
        num_actions   = env.NUM_ACTIONS,
        learning_rate = lr,
        gamma         = gamma,
        epsilon       = epsilon_start,
        epsilon_min   = epsilon_min,
        epsilon_decay = epsilon_decay,
    )

    # ---- Episode logging ----
    episode_logs = []
    print_interval = max(1, episodes // 20)  # print ~20 updates

    for ep in range(1, episodes + 1):
        state      = env.reset()
        state_idx  = env.state_to_index(state)
        total_reward = 0.0
        step_infos   = []

        for _ in range(max_steps):
            action = agent.choose_action(state_idx)
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            agent.learn(state_idx, action, reward, next_state_idx, done)

            state_idx    = next_state_idx
            total_reward += reward
            step_infos.append(info)

            if done:
                break

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Compute episode-level metrics
        ep_metrics = compute_episode_metrics(step_infos)

        episode_logs.append({
            "episode":          ep,
            "total_reward":     round(total_reward, 4),
            "avg_energy_saved": ep_metrics["avg_energy_saved"],
            "avg_comfort_score":ep_metrics["avg_comfort_score"],
            "avg_wait_time":    ep_metrics["avg_wait_time"],
            "epsilon":          round(agent.epsilon, 6),
        })

        if ep % print_interval == 0 or ep == 1:
            print_episode_summary(
                ep, total_reward,
                ep_metrics["avg_energy_saved"],
                ep_metrics["avg_comfort_score"],
                agent.epsilon,
            )

    # ---- Compute final aggregate metrics ----
    last_n = episode_logs[-min(100, len(episode_logs)):]  # last 100 episodes
    final_metrics = {
        "avg_reward":        sum(e["total_reward"]      for e in last_n) / len(last_n),
        "avg_energy_saved":  sum(e["avg_energy_saved"]  for e in last_n) / len(last_n),
        "avg_comfort_score": sum(e["avg_comfort_score"] for e in last_n) / len(last_n),
        "avg_wait_time":     sum(e["avg_wait_time"]     for e in last_n) / len(last_n),
    }

    print_header("Training Complete — Final Metrics (last 100 episodes)")
    print(f"  Avg Reward       : {final_metrics['avg_reward']:.4f}")
    print(f"  Avg Energy Saved : {final_metrics['avg_energy_saved']:.4f}")
    print(f"  Avg Comfort Score: {final_metrics['avg_comfort_score']:.4f}")
    print(f"  Avg Wait Time    : {final_metrics['avg_wait_time']:.4f}\n")

    # ---- Save outputs ----
    run_id = generate_run_id()

    # 1. Model
    agent.save(model_path)

    # 2. CSV
    save_results_csv(csv_path, episode_logs)

    # 3. JSON log
    log_experiment(log_path, run_id, config, final_metrics, notes=notes)

    # 4. Plot
    plot_training_curves(episode_logs, plot_path, title=experiment_name)

    # 5. Print policy snippet
    print("\n[Policy] Sample Q-table rows (first 10 states):")
    print(agent.get_policy_summary())

    print(f"\n[Done] run_id = {run_id}")
    return final_metrics


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)
    train(config, csv_out_override=args.csv_out)
