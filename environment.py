"""
environment.py
==============
Smart Classroom Environment for Reinforcement Learning.

This module simulates a classroom where an RL agent controls:
  - Lights, Fan, AC, Projector

The environment tracks:
  - Occupancy level (0=empty, 1=low, 2=medium, 3=full)
  - Temperature (0=cold, 1=comfortable, 2=hot)
  - Time slot (0=morning, 1=afternoon, 2=evening, 3=night)
  - Device statuses (lights, fan, AC, projector)

Rewards encourage energy savings and student comfort.
"""

import numpy as np
import random


class ClassroomEnvironment:
    """
    Simulates a smart classroom with 4 controllable devices.

    State space: (occupancy, temperature, time_slot, lights, fan, ac, projector)
    Action space: 8 discrete actions (toggle each of 4 devices independently,
                  or more precisely — set a specific device ON/OFF combo)
    """

    # --- State space dimensions ---
    OCCUPANCY_LEVELS = 4    # 0: empty, 1: low (<33%), 2: medium (<66%), 3: full
    TEMPERATURE_LEVELS = 3  # 0: cold (<18°C), 1: comfortable (18-28°C), 2: hot (>28°C)
    TIME_SLOTS = 4          # 0: morning (6-12), 1: afternoon (12-17), 2: evening (17-21), 3: night

    # --- Device actions (binary: ON=1, OFF=0) ---
    # We encode the 4-device combo as a single integer 0-15 (2^4 combinations)
    # For simplicity we reduce to 8 meaningful actions
    NUM_ACTIONS = 16  # all 2^4 combinations of [lights, fan, ac, projector]

    # --- Energy consumption per device (Watts) ---
    ENERGY = {
        "lights": 60,
        "fan": 75,
        "ac": 1500,
        "projector": 300,
    }

    # --- Comfort thresholds ---
    COMFORT_TEMP = 1          # comfortable temperature index
    MIN_OCCUPANCY_FOR_LIGHTS = 1  # lights needed if anyone present

    def __init__(self, max_steps=100, seed=42):
        """
        Initialise the classroom environment.

        Args:
            max_steps (int): Maximum steps per episode.
            seed (int): Random seed for reproducibility.
        """
        self.max_steps = max_steps
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.step_count = 0
        self.state = None
        self.reset()

    # ------------------------------------------------------------------
    # Core RL interface
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset environment to a random initial state.

        Returns:
            tuple: Initial state (occupancy, temperature, time_slot,
                   lights, fan, ac, projector)
        """
        self.step_count = 0

        occupancy    = random.randint(0, self.OCCUPANCY_LEVELS - 1)
        temperature  = random.randint(0, self.TEMPERATURE_LEVELS - 1)
        time_slot    = random.randint(0, self.TIME_SLOTS - 1)
        lights       = random.randint(0, 1)
        fan          = random.randint(0, 1)
        ac           = random.randint(0, 1)
        projector    = random.randint(0, 1)

        self.state = (occupancy, temperature, time_slot, lights, fan, ac, projector)
        return self.state

    def step(self, action):
        """
        Apply an action (device combination) and return next state, reward, done.

        Args:
            action (int): Integer 0-15 encoding [lights, fan, ac, projector] bits.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        assert 0 <= action < self.NUM_ACTIONS, f"Invalid action {action}"

        # Decode action into device settings
        lights_on    = (action >> 3) & 1
        fan_on       = (action >> 2) & 1
        ac_on        = (action >> 1) & 1
        projector_on = (action >> 0) & 1

        occupancy, temperature, time_slot, _, _, _, _ = self.state

        # --- Compute reward ---
        reward, info = self._compute_reward(
            occupancy, temperature, time_slot,
            lights_on, fan_on, ac_on, projector_on
        )

        # --- Transition environment randomly (simulate real classroom dynamics) ---
        occupancy   = self._transition_occupancy(occupancy, time_slot)
        temperature = self._transition_temperature(temperature, ac_on, time_slot)
        time_slot   = (time_slot + (1 if random.random() < 0.1 else 0)) % self.TIME_SLOTS

        self.state = (occupancy, temperature, time_slot,
                      lights_on, fan_on, ac_on, projector_on)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self.state, reward, done, info

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

    def _compute_reward(self, occupancy, temperature, time_slot,
                        lights_on, fan_on, ac_on, projector_on):
        """
        Calculate reward based on energy saving and comfort.

        Reward logic:
          + Energy saved (devices OFF when not needed)
          - Discomfort penalty (lights off when students present, etc.)
          - Waste penalty (AC running in empty room)

        Returns:
            (float, dict): reward value and breakdown dictionary.
        """
        reward = 0.0
        energy_saved = 0.0
        comfort_score = 1.0  # starts perfect, penalised below
        info = {}

        # ---- 1. Lights ----
        if occupancy == 0:
            if lights_on:
                reward     -= 2.0   # wasting electricity, no one here
                energy_saved -= self.ENERGY["lights"] / 100
            else:
                reward     += 1.0   # correctly off
                energy_saved += self.ENERGY["lights"] / 100
        else:
            if lights_on:
                reward     += 1.0   # correctly on for students
            else:
                reward     -= 3.0   # students in dark — big discomfort
                comfort_score -= 0.3

        # ---- 2. Fan ----
        if occupancy == 0:
            if fan_on:
                reward     -= 1.5
                energy_saved -= self.ENERGY["fan"] / 100
            else:
                reward     += 0.5
                energy_saved += self.ENERGY["fan"] / 100
        else:
            if temperature == 2 and fan_on:   # hot, fan helps
                reward     += 1.0
            elif temperature == 0 and fan_on:  # cold, fan hurts
                reward     -= 1.0
                comfort_score -= 0.2
            elif not fan_on and temperature == 2:
                reward     -= 0.5
                comfort_score -= 0.1

        # ---- 3. AC ----
        if occupancy == 0:
            if ac_on:
                reward     -= 5.0   # AC is expensive — heavy penalty
                energy_saved -= self.ENERGY["ac"] / 100
            else:
                reward     += 2.0
                energy_saved += self.ENERGY["ac"] / 100
        else:
            if temperature == 2 and ac_on:    # hot + students: AC correct
                reward     += 2.0
                comfort_score += 0.1
            elif temperature == 0 and ac_on:  # cold + AC: wasteful & uncomfortable
                reward     -= 3.0
                comfort_score -= 0.3
                energy_saved -= self.ENERGY["ac"] / 100
            elif temperature == 1 and ac_on:  # comfortable temp, AC wasteful
                reward     -= 2.0
                energy_saved -= self.ENERGY["ac"] / 100
            elif temperature == 2 and not ac_on:
                reward     -= 1.0
                comfort_score -= 0.2

        # ---- 4. Projector ----
        # Projector needed only during class times (morning/afternoon) with students
        class_time = time_slot in [0, 1]
        if occupancy >= 2 and class_time:
            if projector_on:
                reward     += 1.5   # active class, projector useful
            else:
                reward     -= 0.5   # students might need it
                comfort_score -= 0.1
        else:
            if projector_on:
                reward     -= 1.0   # no class, projector wasting energy
                energy_saved -= self.ENERGY["projector"] / 100
            else:
                reward     += 0.5
                energy_saved += self.ENERGY["projector"] / 100

        # ---- Comfort score clamp ----
        comfort_score = max(0.0, min(1.0, comfort_score))

        # ---- Wait time penalty (proxy: time devices take to reach desired state) ----
        # Modelled as penalty when wrong devices are on (immediate response assumed)
        wait_time = max(0, -reward * 0.05)  # proportional to bad decisions

        info = {
            "energy_saved":   round(energy_saved, 3),
            "comfort_score":  round(comfort_score, 3),
            "wait_time":      round(wait_time, 3),
            "devices": {
                "lights":    lights_on,
                "fan":       fan_on,
                "ac":        ac_on,
                "projector": projector_on,
            }
        }

        return round(reward, 3), info

    # ------------------------------------------------------------------
    # State transition helpers
    # ------------------------------------------------------------------

    def _transition_occupancy(self, occupancy, time_slot):
        """Randomly shift occupancy, biased by time of day."""
        if time_slot == 3:          # night — likely empty
            probs = [0.7, 0.2, 0.08, 0.02]
        elif time_slot in [0, 1]:   # morning/afternoon — likely full
            probs = [0.05, 0.2, 0.35, 0.4]
        else:                       # evening
            probs = [0.2, 0.3, 0.3, 0.2]
        return np.random.choice(self.OCCUPANCY_LEVELS, p=probs)

    def _transition_temperature(self, temperature, ac_on, time_slot):
        """Temperature shifts based on AC usage and time of day."""
        if ac_on:
            # AC cools: shift toward comfortable/cold
            if temperature == 2:
                return np.random.choice([1, 2], p=[0.8, 0.2])
            elif temperature == 1:
                return np.random.choice([0, 1], p=[0.3, 0.7])
            else:
                return 0
        else:
            # No AC: afternoon/evening heats up
            if time_slot == 1 and temperature < 2:
                return min(temperature + (1 if random.random() < 0.3 else 0), 2)
            return temperature

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_state_size(self):
        """Return flat state space size (for Q-table indexing)."""
        return (self.OCCUPANCY_LEVELS * self.TEMPERATURE_LEVELS *
                self.TIME_SLOTS * 2 * 2 * 2 * 2)  # 4 binary devices

    def state_to_index(self, state):
        """
        Convert state tuple to a single integer index for Q-table lookup.

        Args:
            state (tuple): (occupancy, temperature, time_slot, l, f, ac, proj)

        Returns:
            int: Flat index into Q-table.
        """
        occ, temp, ts, l, f, ac, pr = state
        idx = (occ * self.TEMPERATURE_LEVELS * self.TIME_SLOTS * 2 * 2 * 2 * 2 +
               temp * self.TIME_SLOTS * 2 * 2 * 2 * 2 +
               ts   * 2 * 2 * 2 * 2 +
               l    * 2 * 2 * 2 +
               f    * 2 * 2 +
               ac   * 2 +
               pr)
        return idx

    def render(self, state=None):
        """Print a human-readable representation of the current state."""
        s = state if state else self.state
        occ, temp, ts, l, f, ac, pr = s
        occ_map  = {0: "Empty", 1: "Low", 2: "Medium", 3: "Full"}
        temp_map = {0: "Cold", 1: "Comfortable", 2: "Hot"}
        time_map = {0: "Morning", 1: "Afternoon", 2: "Evening", 3: "Night"}
        print(f"  Occupancy : {occ_map[occ]}")
        print(f"  Temp      : {temp_map[temp]}")
        print(f"  Time Slot : {time_map[ts]}")
        print(f"  Lights    : {'ON' if l  else 'OFF'}")
        print(f"  Fan       : {'ON' if f  else 'OFF'}")
        print(f"  AC        : {'ON' if ac else 'OFF'}")
        print(f"  Projector : {'ON' if pr else 'OFF'}")
