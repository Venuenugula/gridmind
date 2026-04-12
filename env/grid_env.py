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

        # _state avoids shadowing the state() method
        self._state = self._init_state()

    def _init_state(self) -> Observation:
        if self.task == "grid_crisis":
            # Elevated demand, reduced renewables, tight market — all pressure at once
            return Observation(
                demand=110.0,
                renewable_supply=45.0,
                thermal_capacity=120.0,
                battery_storage=self.battery,
                price_market=7.0,
                outage_risk=0.15,
                weather_factor=0.75,
                time_of_day=10,
                day_of_week=5,
            )
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

    def state(self) -> Observation:
        return self._state

    async def reset(self):
        self.__init__(self.task)
        return StepResult(observation=self._state, reward=0.0, done=False, info={})

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
            self._state.renewable_supply
            + action.use_thermal
            + action.buy_power
            + discharge
        )

        demand = self._state.demand

        # ---- TASK-SPECIFIC DEMAND STRESS ----
        if self.task == "peak_load":
            demand *= 1.15
        elif self.task == "grid_crisis":
            demand *= 1.15

        demand_met = min(supply, demand)
        excess = max(0.0, supply - demand)

        # ---- MARKET DYNAMICS ----
        imbalance = abs(supply - demand)
        market_noise = random.uniform(-0.5, 0.5)

        # Market shocks: random price spikes in crisis/peak scenarios
        shock = 0.0
        if self.task == "grid_crisis" and random.random() < 0.2:
            shock = random.uniform(5.0, 15.0)

        price_market = max(1.0, 5 + imbalance * 0.05 + market_noise + shock)

        cost = action.buy_power * price_market + action.use_thermal * 6

        renewable_ratio = self._state.renewable_supply / (supply + 1e-6)
        outage_risk = min(1.0, imbalance / 120)

        # ---- CONTROLLED CASCADE ----
        if outage_risk > 0.85:
            self.metrics["outage_events"] += 1
            demand *= 1.1

        # ---- REWARD FUNCTION ----
        reward = 0.0
        reward += (demand_met / demand) * 0.4
        reward += (1 - cost / 1000) * 0.2
        reward += renewable_ratio * 0.15
        reward += (self.battery / BATTERY_CAP) * 0.1
        reward -= outage_risk * 0.2
        reward -= (excess / demand) * 0.15
        reward -= ramp_penalty

        reward = float(max(min(reward, 1.0), -1.0))

        # ---- METRICS UPDATE ----
        self.metrics["demand_satisfaction"] += demand_met / demand
        self.metrics["cost_ratio"] += cost / 1000
        self.metrics["renewable_usage"] += renewable_ratio
        self.metrics["stability"] -= outage_risk

        # ---- RENEWABLE COLLAPSE (renewable_failure + grid_crisis) ----
        if self.task == "renewable_failure":
            if self.step_count >= 4:
                self._state.renewable_supply *= 0.3
            demand *= 1.15
        elif self.task == "grid_crisis":
            # Supply degrades from step 6 onward (slower than renewable_failure)
            if self.step_count >= 6:
                self._state.renewable_supply *= 0.55

        # ---- DEMAND PATTERN ----
        peak_factor = 1.2 if 18 <= self._state.time_of_day <= 22 else 0.9
        new_demand = max(50.0, demand * peak_factor)

        # ---- STATE UPDATE ----
        self._state = Observation(
            demand=new_demand,
            renewable_supply=max(
                10.0,
                self._state.renewable_supply * self._state.weather_factor,
            ),
            thermal_capacity=self._state.thermal_capacity,
            battery_storage=self.battery,
            price_market=price_market,
            outage_risk=outage_risk,
            weather_factor=random.uniform(0.7, 1.0),
            time_of_day=(self._state.time_of_day + 1) % 24,
            day_of_week=self._state.day_of_week,
        )

        done = self.step_count >= self.max_steps

        return StepResult(
            observation=self._state,
            reward=reward,
            done=done,
            info={"metrics": self.metrics, "task": self.task},
        )

    async def close(self):
        return
