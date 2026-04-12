---
title: GridMind v3
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_file: server.app:app
pinned: false
---

# ⚡ GridMind v3
### Benchmark Environment for Autonomous Electricity Grid Optimization

---

## Overview

GridMind v3 is an OpenEnv-compliant reinforcement learning environment that simulates real-world electricity grid operations under dynamic demand, renewable variability, and market-driven pricing.

Unlike toy environments, GridMind models decision-making challenges faced by actual grid operators:

- Demand–supply balancing under uncertainty
- Cost optimization across multiple energy sources
- Renewable energy intermittency
- Battery storage temporal trade-offs
- Market price volatility and shock events
- Cascading failure risk

---

## Why This Matters

Modern power grids face increasing complexity due to:

- Renewable energy intermittency (solar/wind variability)
- Demand spikes during peak hours
- Market-driven electricity pricing
- Risk of cascading failures and outages

GridMind captures these challenges in a controlled, reproducible, LLM-compatible simulation.

---

## Tasks

GridMind features four tasks across a progressive difficulty spectrum:

| Task | Scenario | Difficulty | Heuristic Score |
|------|----------|------------|-----------------|
| `stable_demand` | Predictable conditions — baseline stability | Easy | ~0.60 |
| `peak_load` | High demand spikes requiring rapid multi-source response | Medium | ~0.34 |
| `renewable_failure` | Renewable supply collapse + demand stress | Hard | ~0.43 |
| `grid_crisis` | Peak demand + renewable collapse + market price shocks | Expert | ~0.28 |

LLM agents with structured reasoning outperform the heuristic baseline on all tasks.

---

## Observation Space

| Field | Description |
|-------|-------------|
| `demand` | Current grid demand (MW) |
| `renewable_supply` | Available renewable energy (MW) |
| `thermal_capacity` | Maximum thermal generation capacity (MW) |
| `battery_storage` | Current battery charge (MWh, max 100) |
| `price_market` | Spot electricity price ($/MWh) |
| `outage_risk` | Grid instability index [0, 1] |
| `weather_factor` | Renewable availability multiplier [0.7, 1.0] |
| `time_of_day` | Hour of day [0, 23] |
| `day_of_week` | Day index [0, 6] |

---

## Action Space

| Field | Range | Description |
|-------|-------|-------------|
| `buy_power` | [0, 200] MW | Purchase from electricity market |
| `use_thermal` | [0, 150] MW | Dispatch thermal generation |
| `charge_battery` | [0, 50] MWh | Store energy in battery |
| `discharge_battery` | [0, 50] MWh | Release energy from battery |

---

## Reward Function

The reward balances five competing objectives:

```
reward =
  + 0.40 × (demand_met / demand)          # meeting demand is critical
  + 0.20 × (1 - total_cost / 1000)        # minimize cost
  + 0.15 × (renewable / total_supply)     # prefer renewables
  + 0.10 × (battery_level / 100)          # keep reserves
  - 0.20 × outage_risk                    # avoid instability
  - 0.15 × (excess_supply / demand)       # avoid overproduction
  - ramp_penalty                          # penalize abrupt thermal changes
```

**Cost structure:**
- Thermal generation: $6/MW (fixed)
- Market purchase: `price_market` $/MW (variable, can spike)
- Battery discharge: free

---

## Environment Dynamics

### Market Pricing
Price adapts to supply–demand imbalance plus stochastic noise. The `grid_crisis` task adds random price shocks (spikes of $5–15/MWh) to stress market-aware agents.

### Thermal Ramp Constraints
Sudden changes in thermal output (>30 MW per step) incur a penalty, forcing agents to plan gradual adjustments.

### Battery Storage
Battery introduces intertemporal decisions: discharge is free today but depletes reserves for tomorrow. Agents must manage the charge/discharge trade-off across the 12-step episode.

### Cascade Failures
When `outage_risk > 0.85`, a cascade event increases demand by 10%, creating non-linear pressure that compounds poor decisions.

### Task-Specific Dynamics

- **`peak_load`**: Demand amplified 1.15× each step
- **`renewable_failure`**: Renewables decay to 30% from step 4, demand +15%
- **`grid_crisis`**: Demand +15%, renewables decay to 55% from step 6, random market shocks

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/info` | Environment metadata |
| `GET` | `/tasks` | All task definitions with difficulty |
| `POST` | `/reset?task=<id>` | Reset environment for a given task |
| `POST` | `/step` | Take one action step |
| `GET` | `/state` | Current observation |
| `POST` | `/evaluate?task=<id>` | Run a full heuristic episode |

### Example: Reset + Step

```bash
# Reset for peak_load task
curl -X POST "http://localhost:7860/reset?task=peak_load"

# Take a step
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"buy_power": 20, "use_thermal": 60, "charge_battery": 0, "discharge_battery": 15}'
```

---

## LLM Integration

The inference script (`inference.py`) uses the LiteLLM proxy to call the configured model at every decision step.

**Key design choices:**
- System prompt includes the exact reward formula — the LLM can reason analytically about which action maximises reward
- Chain-of-thought reasoning is encouraged; JSON action is parsed from the final line
- Two-attempt retry before falling back to a greedy heuristic
- Structured observation prompt with computed deficit, cost breakdown, and risk labels

```python
# Env vars required by inference.py
API_BASE_URL=<litellm_proxy_url>
API_KEY=<your_key>
MODEL_NAME=<model_id>
```

---

## Running Locally

### Build and Run with Docker

```bash
docker build -t gridmind .
docker run -p 7860:7860 \
  -e API_BASE_URL=<proxy_url> \
  -e API_KEY=<key> \
  -e MODEL_NAME=<model> \
  gridmind
```

### Run Inference

```bash
python inference.py
```

### Run Server Only

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## Architecture

```
Observation → LLM (via LiteLLM proxy) → Action → GridEnv → Reward + Next State
                        ↑
              [reward formula in system prompt]
              [chain-of-thought reasoning]
              [heuristic fallback on parse error]
```

---

## Design Principles

### Realism
Time-dependent demand patterns, market price volatility, battery constraints, and thermal ramp limits mirror real grid operations.

### Multi-Objective Optimization
Agents balance demand satisfaction, cost, renewable usage, battery health, and stability simultaneously — no single-objective shortcut exists.

### LLM-Compatible
Observations are structured as natural-language-readable state descriptions. The reward formula is embedded in the system prompt for analytical reasoning.

### Progressive Difficulty
Four tasks from easy to expert provide a clear evaluation spectrum. Each task stresses a different aspect of grid management.

---

## Evaluation

Each task runs for 12 steps. Score is the mean per-step reward:

```
score = mean(rewards)  ∈ [-1.0, 1.0]
success = score ≥ 0.3
```

Episode logs follow the format:
```
[START] task=stable_demand env=gridmind model=<model>
[STEP] step=1 action={...} reward=0.65 done=false error=null
[END] success=true steps=12 score=0.597 rewards=0.70,0.65,...
```

---

## What Makes GridMind Different

Most RL environments are:
- Single-objective (minimize cost OR maximize stability)
- Static (no market dynamics)
- Toy-scale (no realistic constraints)

GridMind is:
- Multi-objective with competing trade-offs
- Dynamic with stochastic market prices and weather
- Realistic with ramp constraints, battery physics, and cascade failures
- LLM-native with reward formula embedded in agent context
