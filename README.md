---
title: GridMind v3
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_file: server.py
pinned: false
---

# 🚀 GridMind v3  
### A Benchmark Environment for Autonomous Energy Grid Optimization under Uncertainty

---

## 🧠 Overview

GridMind v3 is a reinforcement learning environment designed to simulate real-world electricity grid operations under dynamic demand, renewable variability, and market-driven pricing.

Unlike toy environments, GridMind models decision-making challenges faced by real grid operators, including:
- Demand–supply balancing  
- Cost optimization under market fluctuations  
- Renewable uncertainty  
- System stability under stress  

This environment is intended as a benchmark for evaluating intelligent agents in safety-critical infrastructure systems.

---

## ⚡ Why This Matters

Modern power grids (especially in regions like India) face increasing complexity due to:

- Renewable energy intermittency (solar/wind variability)  
- Demand spikes during peak hours  
- Market-driven electricity pricing  
- Risk of cascading failures and outages  

GridMind captures these challenges in a controlled, reproducible simulation.

---

## 🎯 Core Design Philosophy

GridMind is built around three principles:

### 1. Realism
- Time-dependent demand patterns  
- Market price fluctuations  
- Battery storage constraints  
- Thermal ramp limitations  

### 2. Multi-Objective Optimization
Agents must balance:
- Demand satisfaction  
- Cost efficiency  
- Renewable usage  
- Grid stability  

### 3. Progressive Difficulty

| Task | Scenario | Difficulty |
|------|--------|----------|
| Stable Demand | Predictable conditions | Easy |
| Peak Load Crisis | High demand spikes | Medium |
| Renewable Failure | Supply collapse + stress | Hard |

---

## 🧩 Environment Design

### Observation Space
- Demand  
- Renewable supply  
- Thermal capacity  
- Battery storage  
- Market price  
- Outage risk  
- Time features  

### Action Space
- Buy power from market  
- Dispatch thermal generation  
- Charge battery  
- Discharge battery  

---

## 🏆 Reward Function

The reward balances:

- Demand satisfaction  
- Cost efficiency  
- Renewable usage  
- Battery utilization  
- Penalties for outage risk, overproduction, and ramp violations  

This provides continuous feedback across the full episode.

---

## 🔥 Key Features

### Market Dynamics
Electricity price adapts based on supply–demand imbalance.

### Cascading Failures
Poor decisions increase outage risk and system stress.

### Energy Storage Modeling
Battery introduces temporal decision-making.

### Deterministic Evaluation
- Fixed seeds  
- Reproducible results  

---

## 🧪 Tasks & Evaluation

Each task includes:
- Defined objective  
- Deterministic grading  
- Score range [0, 1]

### Baseline Performance

| Task | Score |
|------|------|
| Stable Demand | ~0.58 |
| Peak Load | ~0.35 |
| Renewable Failure | ~0.43 |

This shows:
- Strong performance in stable scenarios  
- Graceful degradation under stress  
- Clear differentiation across tasks  

---

## 🏗️ System Architecture

Observation → Action → Environment → Reward → Next State

---

## 🚀 Running the Environment

### Build Docker

docker build -t gridmind .

### Run

docker run -p 7860:7860 gridmind

### Local Test

python inference.py

---

## 🤖 Baseline Agent

A heuristic agent that:
- Responds to demand deficits  
- Uses price signals  
- Accounts for outage risk  

This acts as a reference benchmark.

---

## 📦 OpenEnv Compliance

- Typed Pydantic models  
- reset(), step(), state()  
- Deterministic grading  
- Dockerized environment  
- Hugging Face Space compatible  

---

## 🔬 Use Cases

- Evaluate LLM-based agents  
- Study multi-objective RL  
- Benchmark robustness  
- Simulate infrastructure decision systems  

---

## 🏆 What Makes This Different

Most environments:
- Simplified  
- Static  
- Game-like  

GridMind:
- Real-world grounded  
- Dynamic and stochastic  
- Safety-critical modeling  

---

## 📌 Final Note

GridMind is not just an environment — it is a benchmark for intelligent decision-making in critical infrastructure systems.

---

## 🏁 Conclusion

This project demonstrates:
- Strong system design  
- Real-world relevance  
- Robust evaluation framework  

Designed for advancing AI in infrastructure and decision intelligence.