# 🤖 AI2 – Stochastic Grid World & MDP Controller

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue" />
  <img src="https://img.shields.io/badge/AI-Markov%20Decision%20Processes-purple" />
  <img src="https://img.shields.io/badge/Algorithm-Value%20Iteration-orange" />
  <img src="https://img.shields.io/badge/Environment-Stochastic%20Grid-success" />
</p>

This project is a direct continuation of the KobanGan (Assignment 1) puzzle, extending a deterministic grid environment into a stochastic setting.

An intelligent controller was implemented using **Markov Decision Processes (MDP)** and **Value Iteration**, enabling optimal decision-making under uncertainty.

---

##  Problem Description

The environment is a grid-based pressure-plate puzzle involving:
- Doors and keys
- Pressure plates affecting the environment
- Stochastic action outcomes

Unlike the deterministic version, actions may have probabilistic transitions, requiring planning under uncertainty.

---

##  AI Techniques Used

- 📌 **Markov Decision Processes (MDP)**
- 🔁 **Value Iteration**
- 🎯 Policy extraction from value functions
- 🌫 Handling stochastic transitions and rewards

---

##  Hybrid Planning Approach

To guide learning in the stochastic environment, the project integrates:

- ⭐ A **deterministic A\*** path (from Assignment 1)
- 🧠 Used as a **guiding signal / reward bias**
- 🎯 Improves convergence and policy quality

This hybrid approach combines **classical planning** with **MDP-based optimization**.

---

## ⚙️ System Design

- Explicit state representation
- Transition probability modeling
- Reward function design
- Iterative value updates
- Action selection via learned policy

---

##  Project Structure

```bash
.
├── ex1.py              # Deterministic agent (Assignment 1)
├── ex2.py              # MDP-based stochastic controller
├── pressure_plate.py   # Environment definition
├── search.py           # Search utilities
├── utils.py            # Helper functions
└── README.md
