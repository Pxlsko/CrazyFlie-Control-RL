# CrazyFlie-Control-RL

A modular framework for reinforcement learning-based control of CrazyFlie drones in PyBullet environments.

---

## 📦 Project Structure

```
gym_pybullet_drones/
│
├── assets/                # Drone models, trajectories, trained agent (to be added)
├── envs/                  # Custom Gymnasium environments
├── control/               # Control algorithms (PID, RL, etc.)
├── utils/                 # Logging, plotting, and helper functions
├── examples/              # Training and evaluation scripts
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```sh
git clone https://github.com/Pxlsko/CrazyFlie-Control-RL.git
cd CrazyFlie-Control-RL
pip install -e .
```

### 2. Training an Agent

To train a reinforcement learning agent:

```sh
python gym_pybullet_drones/examples/learn.py
```

- Training parameters can be adjusted in `learn.py`.
- The trained agent will be saved in the `assets/` folder (to be added).

### 3. Evaluating a Trained Agent

To evaluate a trained policy:

```sh
python gym_pybullet_drones/examples/evaluation.py
```

- The script loads the agent from `assets/` and runs it in the selected environment.
- Results and plots are saved in the `results/Trajectoryresults/` folder.

---

## 🧩 Environments

- **RandomPointAviary**: Reach a random target position.
- **TestRandomPointAviary**: Follow a trajectory or hover at multiple waypoints.
- **ConstantPerturbation**: Reject a constant external force.
- **ImpactPerturbation**: Reject an impact perturbation.

Each environment is customizable via parameters in the example scripts.

---

## 🤖 Trained Agent Policy (To Be Added)

The trained agent policy will be included in the `assets/` folder as:

- `best_model.zip`: Stable Baselines3 PPO agent.
- `final_model.zip`: Final checkpoint after training.

**Policy Features:**
- Robust to external perturbations.
- Capable of trajectory tracking and hovering.
- Evaluated on multiple scenarios (random points, helix, square, etc.).

**How to Use:**
- Place the trained model in `assets/`.
- Run `evaluation.py` to visualize and analyze performance.

---

## 📊 Results & Visualization

- Training logs and TensorBoard files are saved in `results/Trajectoryresults/tb/`.
- Evaluation metrics (mean reward, precision at waypoints) are printed and plotted.
- Example plots include position tracking, orientation, and velocity comparisons.

---

## 📝 To Do

- [ ] Add trained agent policy to `assets/`.
- [ ] Update results section with evaluation metrics and plots.
- [ ] Expand documentation for custom environments.

---

## 📚 References

- [PyBullet](https://pybullet.org/)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [CrazyFlie](https://www.bitcraze.io/products/crazyflie-2-1/)

---

## 💡 Contact

For questions or contributions, open an issue or contact [your email].
