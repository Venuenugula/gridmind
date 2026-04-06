import random
import numpy as np
from .models import Observation, Action, StepResult

# ---- CONSTANTS ----
MAX_BUY = 200
MAX_THERMAL = 150
BATTERY_CAP = 100


class GridEnv:
    def __init__(self, task="stable_demand"):
        random.seed(42)
        np.random.seed(42)

        self.task = task
        self.max_steps = 12
        self.step_count = 0

        self.prev_thermal = 0
        self.battery = 50

        self.metrics = {
            "demand_satisfaction": 0.0,
            "cost_ratio": 0.0,
            "renewable_usage": 0.0,
            "stability": 1.0,
            "outage_events": 0,
        }

        self.state = self._init_state()

    def _init_state(self):
        return Observation(
            demand=100.0,
            renewable_supply=60.0,
            thermal_capacity=120.0,
            battery_storage=self.battery,
            price_market=5.0,
            outage_risk=0.1,
            weather_factor=0.8,
            time_of_day=10,
            day_of_week=2,
        )

    async def reset(self):
        self.__init__(self.task)
        return StepResult(observation=self.state, reward=0.0, done=False, info={})

    async def step(self, action: Action):
        self.step_count += 1

        # ---- CLAMP ACTIONS ----
        action.buy_power = max(0, min(action.buy_power, MAX_BUY))
        action.use_thermal = max(0, min(action.use_thermal, MAX_THERMAL))
        action.charge_battery = max(0, action.charge_battery)
        action.discharge_battery = max(0, action.discharge_battery)

        # ---- RAMP CONSTRAINT ----
        ramp_penalty = 0.0
        if abs(action.use_thermal - self.prev_thermal) > 30:
            ramp_penalty = 0.1
        self.prev_thermal = action.use_thermal

        # ---- BATTERY ----
        self.battery = min(BATTERY_CAP, self.battery + action.charge_battery)
        discharge = min(self.battery, action.discharge_battery)
        self.battery -= discharge

        # ---- SUPPLY ----
        supply = (
            self.state.renewable_supply
            + action.use_thermal
            + action.buy_power
            + discharge
        )

        demand = self.state.demand

        # ---- FIX 1: BETTER PEAK LOAD (LESS HARSH) ----
        if self.task == "peak_load":
            demand *= 1.15   # was 1.3 → too aggressive

        demand_met = min(supply, demand)
        excess = max(0.0, supply - demand)

        # ---- MARKET DYNAMICS ----
        imbalance = abs(supply - demand)
        market_noise = random.uniform(-0.5, 0.5)  # reduced volatility
        price_market = max(1.0, 5 + imbalance * 0.05 + market_noise)

        cost = action.buy_power * price_market + action.use_thermal * 6

        renewable_ratio = self.state.renewable_supply / (supply + 1e-6)
        outage_risk = min(1.0, imbalance / 120)  # smoother risk curve

        # ---- FIX 2: CONTROLLED CASCADE ----
        if outage_risk > 0.85:
            self.metrics["outage_events"] += 1
            demand *= 1.1  # reduced cascade (was 1.2)

        # ---- REWARD FUNCTION ----
        reward = 0.0

        reward += (demand_met / demand) * 0.4
        reward += (1 - cost / 1000) * 0.2
        reward += renewable_ratio * 0.15
        reward += (self.battery / BATTERY_CAP) * 0.1

        # ---- FIX 3: SMOOTHER PENALTIES ----
        reward -= outage_risk * 0.2      # reduced from 0.25
        reward -= (excess / demand) * 0.15
        reward -= ramp_penalty

        reward = float(max(min(reward, 1.0), -1.0))

        # ---- METRICS UPDATE ----
        self.metrics["demand_satisfaction"] += demand_met / demand
        self.metrics["cost_ratio"] += cost / 1000
        self.metrics["renewable_usage"] += renewable_ratio
        self.metrics["stability"] -= outage_risk

        # ---- FIX 4: HARDER RENEWABLE FAILURE ----
        if self.task == "renewable_failure":
            if self.step_count >= 4:
                self.state.renewable_supply *= 0.3

            demand *= 1.15

        # ---- DEMAND PATTERN ----
        peak_factor = 1.2 if 18 <= self.state.time_of_day <= 22 else 0.9
        new_demand = max(50.0, demand * peak_factor)

        # ---- STATE UPDATE ----
        self.state = Observation(
            demand=new_demand,
            renewable_supply=max(
                10.0,
                self.state.renewable_supply * self.state.weather_factor,
            ),
            thermal_capacity=self.state.thermal_capacity,
            battery_storage=self.battery,
            price_market=price_market,
            outage_risk=outage_risk,
            weather_factor=random.uniform(0.7, 1.0),
            time_of_day=(self.state.time_of_day + 1) % 24,
            day_of_week=self.state.day_of_week,
        )

        done = self.step_count >= self.max_steps

        return StepResult(
            observation=self.state,
            reward=reward,
            done=done,
            info={"metrics": self.metrics},
        )

    def state(self):
        return self.state

    async def close(self):
        return