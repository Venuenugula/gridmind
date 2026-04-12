from fastapi import FastAPI, Query
from env.grid_env import GridEnv
from env.models import Action
from env.tasks import TASKS

app = FastAPI(title="GridMind", version="3.0")

# Active environment instance (task can be switched via /reset?task=...)
env = GridEnv()


@app.get("/")
def home():
    return {"status": "GridMind running", "version": "3.0", "env_id": "gridmind"}


@app.get("/info")
def info():
    return {
        "env_id": "gridmind",
        "name": "GridMind",
        "version": "3.0",
        "description": (
            "Benchmark environment for autonomous electricity grid optimization "
            "under dynamic demand, renewable variability, and market-driven pricing."
        ),
        "tasks": list(TASKS.keys()),
        "observation_space": [
            "demand", "renewable_supply", "thermal_capacity", "battery_storage",
            "price_market", "outage_risk", "weather_factor", "time_of_day", "day_of_week",
        ],
        "action_space": ["buy_power", "use_thermal", "charge_battery", "discharge_battery"],
    }


@app.get("/tasks")
def tasks():
    return TASKS


@app.post("/reset")
async def reset(task: str = Query(default="stable_demand")):
    global env
    env = GridEnv(task=task)
    return (await env.reset()).model_dump()


@app.post("/step")
async def step(action: dict):
    return (await env.step(Action(**action))).model_dump()


@app.get("/state")
def state():
    return env.state().model_dump()


@app.post("/evaluate")
async def evaluate(task: str = Query(default="stable_demand"), max_steps: int = Query(default=12)):
    """
    Run a full heuristic episode and return per-step rewards and final score.
    Useful for sanity-checking environment behaviour before LLM inference.
    """
    eval_env = GridEnv(task=task)
    result = await eval_env.reset()

    rewards = []
    step_log = []
    steps = 0

    while not result.done and steps < max_steps:
        obs = result.observation
        # Built-in greedy heuristic: discharge battery first, then thermal, then buy
        deficit = max(0.0, obs.demand - obs.renewable_supply)
        battery_use = min(obs.battery_storage, deficit, 50.0)
        remaining = max(0.0, deficit - battery_use)
        thermal = min(remaining * 0.7, 100.0)
        buy = min(max(0.0, remaining - thermal), 80.0)

        action = Action(
            buy_power=buy,
            use_thermal=thermal,
            charge_battery=10.0 if obs.renewable_supply > obs.demand else 0.0,
            discharge_battery=battery_use,
        )
        result = await eval_env.step(action)
        rewards.append(result.reward)
        step_log.append({
            "step": steps + 1,
            "action": action.model_dump(),
            "reward": result.reward,
            "outage_risk": result.observation.outage_risk,
            "demand": result.observation.demand,
        })
        steps += 1

    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = max(0.0, min(score, 1.0))
    await eval_env.close()

    return {
        "task": task,
        "steps": steps,
        "score": round(score, 4),
        "success": score >= 0.3,
        "rewards": [round(r, 4) for r in rewards],
        "step_log": step_log,
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
