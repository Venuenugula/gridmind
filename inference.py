import asyncio
import json
import os
import re
from typing import List

from env.grid_env import GridEnv
from env.models import Action
from openai import OpenAI

# ---- CONFIG (required env vars) ----
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

TASKS = ["stable_demand", "peak_load", "renewable_failure", "grid_crisis"]
MAX_STEPS = 12
SUCCESS_THRESHOLD = 0.3

SYSTEM_PROMPT = """You are an expert AI electricity grid operator. Your mission is to keep the grid stable and efficient.

## Reward Formula (maximize your total score):
  +0.40 × (supply_met / demand)           — meeting demand is critical
  +0.20 × (1 - total_cost / 1000)         — minimize costs
  +0.15 × (renewable_supply / total_supply) — maximize renewable ratio
  +0.10 × (battery_level / 100)            — keep battery reserves
  -0.20 × outage_risk                      — avoid instability
  -0.15 × (excess_supply / demand)         — avoid overproduction

## Cost structure:
  - Thermal generation: $6/MW (fixed cost)
  - Market purchase: price_market $/MW (variable, can spike)
  - Battery discharge: FREE (use stored energy)
  - Battery charge: reduces future costs

## Decision strategy:
  1. Calculate deficit = demand - renewable_supply
  2. Discharge battery first (it's free and reduces outage risk)
  3. Use thermal for remaining deficit (cheap at $6/MW)
  4. Buy from market only if still short (expensive, especially when price > $8)
  5. Charge battery only if supply > demand (store excess renewable)
  6. If outage_risk > 0.7: prioritize meeting demand above all else

## Output format:
Think step by step, then end your response with a JSON object on the last line:
{"buy_power": <float>, "use_thermal": <float>, "charge_battery": <float>, "discharge_battery": <float>}

Constraints: buy_power ∈ [0,200], use_thermal ∈ [0,150], charge/discharge ∈ [0,50]"""


def build_prompt(obs, step: int, task: str) -> str:
    deficit = max(0.0, obs.demand - obs.renewable_supply)
    risk_label = "CRITICAL" if obs.outage_risk > 0.7 else ("HIGH" if obs.outage_risk > 0.4 else "LOW")
    thermal_cost = 6.0
    market_cost = obs.price_market
    return (
        f"=== Grid State: Step {step} | Task: {task} ===\n"
        f"Demand:           {obs.demand:.1f} MW\n"
        f"Renewable Supply: {obs.renewable_supply:.1f} MW\n"
        f"Supply Deficit:   {deficit:.1f} MW  ← must cover this\n"
        f"Thermal Capacity: {obs.thermal_capacity:.1f} MW (costs ${thermal_cost}/MW)\n"
        f"Battery Storage:  {obs.battery_storage:.1f}/100 MWh (discharge is FREE)\n"
        f"Market Price:     ${market_cost:.2f}/MWh\n"
        f"Outage Risk:      {obs.outage_risk:.2f} ({risk_label})\n"
        f"Time of Day:      {obs.time_of_day}:00\n"
        f"Weather Factor:   {obs.weather_factor:.2f}\n\n"
        f"Decide actions to cover {deficit:.1f} MW deficit. "
        f"Battery has {obs.battery_storage:.1f} MWh available for free discharge."
    )


def parse_action(text: str) -> dict | None:
    """Extract JSON action from LLM response (handles chain-of-thought prefix)."""
    # Try the last line first (CoT puts JSON at end)
    lines = text.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    # Full text parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Regex fallback
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def heuristic_fallback(obs) -> Action:
    """Heuristic policy — fallback when LLM response cannot be parsed."""
    deficit = max(0.0, obs.demand - obs.renewable_supply)
    # Use battery first (free), then thermal, then market
    battery_use = min(obs.battery_storage, deficit, 50.0)
    remaining = max(0.0, deficit - battery_use)
    if obs.outage_risk > 0.7:
        thermal = min(remaining, 100.0)
        buy = min(max(0.0, remaining - thermal), 100.0)
    elif obs.price_market < 6.0:
        # Market is cheaper than thermal — buy more
        buy = min(remaining * 0.6, 100.0)
        thermal = min(remaining * 0.4, 80.0)
    else:
        thermal = min(remaining * 0.7, 100.0)
        buy = min(remaining * 0.3, 80.0)
    charge = 10.0 if obs.renewable_supply > obs.demand and obs.battery_storage < 80 else 0.0
    return Action(
        buy_power=buy,
        use_thermal=thermal,
        charge_battery=charge,
        discharge_battery=battery_use,
    )


def llm_decide(obs, step: int, task: str) -> Action:
    """Call the LLM via LiteLLM proxy to decide the grid action."""
    prompt = build_prompt(obs, step, task)
    last_error = None

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            text = response.choices[0].message.content or ""
            print(f"[LLM] step={step} attempt={attempt+1} response={text[:300]}", flush=True)
            parsed = parse_action(text)
            if parsed:
                return Action(
                    buy_power=float(parsed.get("buy_power", 0.0)),
                    use_thermal=float(parsed.get("use_thermal", 0.0)),
                    charge_battery=float(parsed.get("charge_battery", 0.0)),
                    discharge_battery=float(parsed.get("discharge_battery", 0.0)),
                )
            print(f"[LLM] step={step} attempt={attempt+1} parse_failed", flush=True)
        except Exception as e:
            last_error = e
            print(f"[LLM] step={step} attempt={attempt+1} error={e}", flush=True)

    print(f"[LLM] step={step} using heuristic fallback (last_error={last_error})", flush=True)
    return heuristic_fallback(obs)


def log_start(task: str):
    print(f"[START] task={task} env=gridmind model={MODEL_NAME}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: str | None = None):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
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

            # LLM decides every action (heuristic is only a parse-failure fallback)
            action = llm_decide(obs, step=steps + 1, task=task)

            result = await env.step(action)
            reward = result.reward or 0.0
            done = result.done
            steps += 1
            rewards.append(reward)

            log_step(step=steps, action=action.model_dump(), reward=reward, done=done)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
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
