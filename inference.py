import asyncio
import os
from typing import List

from env.grid_env import GridEnv
from env.models import Action

# ---- CONFIG (required env vars) ----
API_BASE_URL = os.getenv("API_BASE_URL", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline")
HF_TOKEN = os.getenv("HF_TOKEN", "none")

TASKS = ["stable_demand", "peak_load", "renewable_failure"]
MAX_STEPS = 12
SUCCESS_THRESHOLD = 0.3


def log_start(task: str):
    print(f"[START] task={task} env=gridmind model={MODEL_NAME}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: str | None):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def smart_policy(obs):
    deficit = obs.demand - obs.renewable_supply

    # clamp deficit
    deficit = max(0, deficit)

    # risk-aware control
    if obs.outage_risk > 0.7:
        buy_power = min(deficit, 100)
        thermal = min(deficit * 0.8, 100)

    elif obs.price_market < 6:
        buy_power = min(deficit * 0.6, 100)
        thermal = min(deficit * 0.4, 100)

    else:
        buy_power = min(deficit * 0.3, 100)
        thermal = min(deficit * 0.7, 100)

    return Action(
        buy_power=buy_power,
        use_thermal=thermal,
        charge_battery=10 if obs.renewable_supply > obs.demand else 0,
        discharge_battery=min(15, obs.battery_storage),
    )

async def run_task(task: str):
    env = GridEnv(task=task)

    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0

    log_start(task)

    try:
        result = await env.reset()

        while not result.done and steps < MAX_STEPS:
            obs = result.observation

            action = smart_policy(obs)

            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done

            steps += 1
            rewards.append(reward)

            log_step(
                step=steps,
                action=action.model_dump(),
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        # normalize score
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(score, 1.0))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        # MUST NOT crash silently
        log_step(steps, {}, 0.0, True, str(e))

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(success, steps, score, rewards)


async def main():
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())