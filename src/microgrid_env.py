import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class MicrogridEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface for Microgrid Battery Management.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data: pd.DataFrame, max_cap_kwh: float = 10.0, max_rw_kw: float = 2.5):
        super(MicrogridEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.max_steps = len(self.data) - 1
        self.current_step = 0
        
        # Battery Constraints
        self.max_cap_kwh = max_cap_kwh
        self.max_rw_kw = max_rw_kw # Max charge/discharge per hour (since step is 1 hr)
        self.soc_kwh = 0.0 # Initial State of Charge
        
        # Actions: 0 = Discharge max, 1 = Idle, 2 = Charge max
        self.action_space = spaces.Discrete(3)
        
        # Observations: [SOC_ratio, Demand, PV, Tariff]
        high = np.array([
            1.0,                     # Max SOC ratio
            self.data["demand_kw"].max() * 1.5,
            self.data["pv_kw"].max() * 1.5,
            self.data["tariff_rate"].max() * 1.5
        ], dtype=np.float32)
        
        low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        row = self.data.iloc[self.current_step]
        return np.array([
            self.soc_kwh / self.max_cap_kwh,
            row["demand_kw"],
            row["pv_kw"],
            row["tariff_rate"]
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {"step": self.current_step, "soc_kwh": self.soc_kwh}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.soc_kwh = 0.0
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        row = self.data.iloc[self.current_step]
        demand = row["demand_kw"]
        pv = row["pv_kw"]
        tariff = row["tariff_rate"]
        
        # Calculate net load before battery
        net_load = demand - pv
        
        # Battery Action
        battery_power = 0.0 # Positive means charging, Negative means discharging
        if action == 0: # Discharge
            # Can only discharge what we have, up to max rate
            battery_power = -min(self.max_rw_kw, self.soc_kwh)
        elif action == 2: # Charge
            # Can only charge what fits, up to max rate
            battery_power = min(self.max_rw_kw, self.max_cap_kwh - self.soc_kwh)
            
        # Update SOC
        self.soc_kwh += battery_power
        
        # Grid interaction: Net Load + Battery Power
        # If > 0, we buy from grid. If < 0, we sell to grid (assume 0 compensation for simplicity, or we clip it)
        grid_load = max(0.0, net_load + battery_power) 
        
        # Cost is what we pay to the grid
        energy_cost = grid_load * tariff
        
        # Degradation Penalty (penalize extreme SOCs)
        soc_ratio = self.soc_kwh / self.max_cap_kwh
        degradation_penalty = 0.0
        if soc_ratio < 0.1 or soc_ratio > 0.9:
            degradation_penalty = 0.05 * tariff # Small penalty
            
        reward = -energy_cost - degradation_penalty
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        obs = self._get_obs()
        print(f"Step: {self.current_step} | SOC: {obs[0]:.2%} | Demand: {obs[1]:.2f} kW | PV: {obs[2]:.2f} kW | Tariff: ${obs[3]:.3f}/kWh")
