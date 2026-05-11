"""
utils.py
========
Utility functions for the Smart Classroom RL project.

Provides:
  - Experiment logging (JSON + CSV)
  - Reward-curve plotting
  - Console pretty-printing helpers
  - Config loading (YAML)
"""

import os
import json
import csv
import datetime
import uuid
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering (no display required)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ===========================================================================
# Config loader
# ===========================================================================

def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to .yaml file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[Config] Loaded: {config_path}")
    return config


# ===========================================================================
# Experiment logging
# ===========================================================================

def generate_run_id():
    """Generate a short unique run identifier."""
    return str(uuid.uuid4())[:8]


def log_experiment(log_path, run_id, config, metrics, notes=""):
    """
    Append an experiment record to log.json.

    Args:
        log_path  (str) : Path to log.json file.
        run_id    (str) : Unique run identifier.
        config    (dict): Hyperparameter config dict.
        metrics   (dict): keys: avg_reward, avg_energy_saved,
                          avg_comfort_score, avg_wait_time
        notes     (str) : Optional free-text notes.
    """
    record = {
        "run_id":               run_id,
        "timestamp":            datetime.datetime.now().isoformat(),
        "episodes":             config.get("episodes", "N/A"),
        "epsilon":              config.get("epsilon_start", "N/A"),
        "learning_rate":        config.get("learning_rate", "N/A"),
        "gamma":                config.get("gamma", "N/A"),
        "average_reward":       round(metrics.get("avg_reward", 0), 4),
        "average_energy_saved": round(metrics.get("avg_energy_saved", 0), 4),
        "average_comfort_score":round(metrics.get("avg_comfort_score", 0), 4),
        "average_wait_time":    round(metrics.get("avg_wait_time", 0), 4),
        "notes":                notes,
    }

    # Load existing log or start fresh
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                all_logs = json.load(f)
            except json.JSONDecodeError:
                all_logs = []
    else:
        all_logs = []

    all_logs.append(record)

    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)

    print(f"[Logger] Experiment logged → {log_path}  (run_id={run_id})")
    return record


def save_results_csv(csv_path, episode_logs):
    """
    Write per-episode metrics to a CSV file.

    Args:
        csv_path     (str) : Output CSV path.
        episode_logs (list): List of dicts with episode-level data.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not episode_logs:
        print("[CSV] No episode logs to save.")
        return

    fieldnames = episode_logs[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_logs)

    print(f"[CSV] Results saved → {csv_path}  ({len(episode_logs)} episodes)")


# ===========================================================================
# Plotting
# ===========================================================================

def plot_training_curves(episode_logs, save_path, title="Training Curves"):
    """
    Generate a 2×2 panel of training metric curves and save to PNG.

    Panels:
      1. Episode reward over time
      2. Moving-average reward (window=50)
      3. Average energy saved per episode
      4. Average comfort score per episode

    Args:
        episode_logs (list): Per-episode dicts.
        save_path    (str) : Output PNG path.
        title        (str) : Figure super-title.
    """
    df = pd.DataFrame(episode_logs)

    window = min(50, max(1, len(df) // 10))  # adaptive smoothing window
    df["smooth_reward"] = df["total_reward"].rolling(window, min_periods=1).mean()

    # ---- Colour palette ----
    colors = {
        "reward":  "#4E9AF1",
        "smooth":  "#FF6B6B",
        "energy":  "#6BCB77",
        "comfort": "#FFD93D",
    }

    fig = plt.figure(figsize=(14, 9), facecolor="#1a1a2e")
    fig.suptitle(title, fontsize=16, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ---- Helper: style axis ----
    def style_ax(ax, xlabel, ylabel, ttl):
        ax.set_facecolor("#16213e")
        ax.set_xlabel(xlabel, color="#aaaacc", fontsize=9)
        ax.set_ylabel(ylabel, color="#aaaacc", fontsize=9)
        ax.set_title(ttl, color="white", fontsize=10, pad=8)
        ax.tick_params(colors="#888888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.grid(True, color="#222244", linewidth=0.5, linestyle="--")

    # Panel 1 – Episode reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["episode"], df["total_reward"],
             color=colors["reward"], alpha=0.45, linewidth=0.8, label="Reward")
    ax1.plot(df["episode"], df["smooth_reward"],
             color=colors["smooth"], linewidth=2.0, label=f"MA-{window}")
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
    style_ax(ax1, "Episode", "Total Reward", "Episode Reward")

    # Panel 2 – Reward distribution histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["total_reward"], bins=30, color=colors["reward"],
             edgecolor="#1a1a2e", alpha=0.85)
    ax2.axvline(df["total_reward"].mean(), color=colors["smooth"],
                linestyle="--", linewidth=1.5, label="Mean")
    ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    style_ax(ax2, "Reward", "Count", "Reward Distribution")

    # Panel 3 – Energy saved
    ax3 = fig.add_subplot(gs[1, 0])
    energy_smooth = df["avg_energy_saved"].rolling(window, min_periods=1).mean()
    ax3.fill_between(df["episode"], energy_smooth, alpha=0.35, color=colors["energy"])
    ax3.plot(df["episode"], energy_smooth, color=colors["energy"], linewidth=1.8)
    style_ax(ax3, "Episode", "Avg Energy Saved", "Energy Saved per Episode")

    # Panel 4 – Comfort score
    ax4 = fig.add_subplot(gs[1, 1])
    comfort_smooth = df["avg_comfort_score"].rolling(window, min_periods=1).mean()
    ax4.fill_between(df["episode"], comfort_smooth, alpha=0.35, color=colors["comfort"])
    ax4.plot(df["episode"], comfort_smooth, color=colors["comfort"], linewidth=1.8)
    ax4.set_ylim(0, 1.1)
    style_ax(ax4, "Episode", "Comfort Score (0–1)", "Student Comfort Score")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


# ===========================================================================
# Console helpers
# ===========================================================================

def print_header(text):
    """Print a formatted section header to console."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_episode_summary(episode, total_reward, avg_energy, avg_comfort, epsilon):
    """Print a one-line training progress update."""
    print(f"  Ep {episode:>5} | Reward: {total_reward:>8.2f} | "
          f"Energy: {avg_energy:>6.3f} | Comfort: {avg_comfort:>5.3f} | "
          f"ε: {epsilon:.4f}")


def compute_episode_metrics(step_infos):
    """
    Aggregate step-level info dicts into episode-level metrics.

    Args:
        step_infos (list): List of info dicts returned by env.step().

    Returns:
        dict: avg_energy_saved, avg_comfort_score, avg_wait_time
    """
    if not step_infos:
        return {"avg_energy_saved": 0, "avg_comfort_score": 0, "avg_wait_time": 0}

    avg_energy  = float(np.mean([i.get("energy_saved",  0) for i in step_infos]))
    avg_comfort = float(np.mean([i.get("comfort_score", 0) for i in step_infos]))
    avg_wait    = float(np.mean([i.get("wait_time",     0) for i in step_infos]))

    return {
        "avg_energy_saved":   round(avg_energy,  4),
        "avg_comfort_score":  round(avg_comfort, 4),
        "avg_wait_time":      round(avg_wait,    4),
    }
