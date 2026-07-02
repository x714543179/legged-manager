# Legged Manager: Manager-Based Isaac Gym Training Code for Legged Robot Locomotion






**Legged Manager** is an Isaac Gym based reinforcement learning framework for legged and wheel-legged robot locomotion.

This repository extends the classic `legged_gym + rsl_rl` training stack with an **IsaacLab-style manager architecture**, making robot tasks easier to organize, customize, debug, and reuse.

The current release focuses on a **Go2W robot locomotion task** with modular managers for actions, commands, observations, rewards, terminations, events, terrain generation, domain randomization, latency simulation, and DreamWaQ policy training.

---

## Highlights

**Manager-based task architecture**
  Task logic is split into reusable managers:

  * `ActionManager`
  * `CommandManager`
  * `ObservationManager`
  * `RewardManager`
  * `TerminationManager`
  * `EventManager`

---

## Supported task

| Task   | Robot | Terrain                  | Policy                     | Status |
| ------ | ----- | ------------------------ | -------------------------- | ------ |
| `go2w` | Go2W  | rough terrain and stairs | DreamWaQ | active |

---

## Installation

### 1. Create environment

```bash
conda create -n legged-manager python=3.8
conda activate legged-manager
```

### 2. Install Isaac Gym
Firstly, **install the PyTorch and CUDA versions required by Isaac Gym environment requirements**.

Then, **download Isaac Gym from NVIDIA, then install the Python package**:

```bash
cd isaacgym/python
pip install -e .
```

Test Isaac Gym:

```bash
cd examples
python 1080_balls_of_solitude.py
```

### 3. Clone this repository

```bash
git clone https://github.com/x714543179/legged-manager.git
cd legged-manager
```

### 4. Install local RSL-RL

```bash
cd rsl_rl-1.0.2
pip install -e .
```

### 5. Install legged_gym

```bash
cd ../legged_gym
pip install -e .
```

---

## Training

Train the Go2W policy:

```bash
cd legged_gym
python legged_gym/scripts/train.py --task=go2w 
```

Useful options:

```bash
python legged_gym/scripts/train.py --task=go2w --num_envs=4096 --max_iterations=10000 --headless
```

---

## Play

Run the latest trained policy:

```bash
cd legged_gym
python legged_gym/scripts/play.py --task=go2w --num_envs=8
```

---

## Logs and checkpoints

Training logs and checkpoints are saved under:

```text
legged_gym/logs/<experiment_name>/<run_name>/
```

---

## Manager architecture

The core design of this repository is a **manager-based Isaac Gym environment architecture**.

Classic `legged_gym` is fast and widely used, but when a task becomes complex, environment logic can easily become tightly coupled. Reward functions, observation construction, randomization, command sampling, reset conditions, and action processing may be mixed inside large environment files, making new tasks harder to modify, debug, and compare.

Legged Manager reorganizes the Isaac Gym training pipeline around **configurable manager terms**. Each part of the environment is handled by an independent manager, and each manager dispatches a group of registered terms defined in the task configuration.

```text
ManagerBasedTask
├── ActionManager
├── CommandManager
├── ObservationManager
├── RewardManager
├── TerminationManager
└── EventManager
```

The manager-based design makes the environment easier to extend in several ways:

* new observation terms can be added without rewriting the whole environment;
* reward terms can be enabled, disabled, replaced, or reweighted from configuration;
* termination conditions can be organized as independent safety and reset checks;
* domain randomization can be written as event terms;
* command generators can be replaced for different locomotion tasks;
* different robot tasks can share the same manager infrastructure;
* ablation studies become cleaner because each term has an explicit configuration entry;
* IsaacLab-style modular environment design can be used inside an Isaac Gym training pipeline.

For example, reward terms can be defined as structured configuration items:

```python
class rewards_manager:
    tracking_lin_vel = ManagerTermCfg(
        func=mdp.tracking_lin_vel,
        scale=3.0,
        env_arg=True,
    )

    tracking_ang_vel = ManagerTermCfg(
        func=mdp.tracking_ang_vel,
        scale=1.5,
        env_arg=True,
    )

    orientation = ManagerTermCfg(
        func=mdp.orientation,
        scale=-2.0,
        env_arg=True,
    )
```

Event terms such as friction randomization can also be configured independently:

```python
class events:
    friction = ManagerTermCfg(
        func=mdp.randomize_friction,
        mode="asset_init",
        env_arg=True,
        params={
            "enabled": True,
            "friction_range": [0.2, 1.25],
        },
    )
```

Observation groups can be organized through the observation manager:

```python
class observations:
    class actor(ObsGroup):
        imu = ManagerTermCfg(func=mdp.imu, env_arg=True)
        command = ManagerTermCfg(func="_obs_commands")
        motor = ManagerTermCfg(func=mdp.motor, env_arg=True)
        dof_pos = ManagerTermCfg(func=mdp.dof_pos, env_arg=True)
        action = ManagerTermCfg(func="_obs_actions")
```

This design is especially useful for legged robot reinforcement learning, where training performance depends on many coupled components such as commands, rewards, observations, reset rules, terrain curriculum, and randomization. By turning these components into manager terms, the codebase becomes easier to maintain and more suitable for rapid research iteration.

---

## Terrain and curriculum

The current terrain configuration supports:

* pyramid slopes;
* random rough slopes;
* stairs up;
* stairs down;
* discrete obstacles;
* curriculum over terrain difficulty;
* height measurement observations for critic input.

The terrain generator can be extended with parkour-style terrain such as gaps, hurdles, beams, corridors, and stepping stones.

---

## Repository structure

```text
legged-manager/
├── legged_gym/
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── base/
│   │   │   └── go2w/
│   │   │       ├── go2w_dreamwaq/
│   │   │       └── mdp/
│   │   ├── managers/
│   │   │   ├── action_manager.py
│   │   │   ├── command_manager.py
│   │   │   ├── event_manager.py
│   │   │   ├── manager_base.py
│   │   │   ├── observation_manager.py
│   │   │   ├── reward_manager.py
│   │   │   └── termination_manager.py
│   │   ├── scripts/
│   │   │   ├── train.py
│   │   │   └── play.py
│   │   ├── terrains/
│   │   └── utils/
│   └── setup.py
├── rsl_rl-1.0.2/
│   └── rsl_rl/
├── .gitignore
├── .gitattributes
└── README.md
```

---

## Acknowledgements

This repository builds on and is inspired by several excellent open-source projects in the legged robot reinforcement learning community.

* [`legged_gym`](https://github.com/leggedrobotics/legged_gym) provides the original Isaac Gym based training framework for legged robot locomotion.
* [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) provides the original GPU-based PPO reinforcement learning implementation.
* [`DreamWaQ`](https://github.com/Manaro-Alpha/DreamWaQ) inspired part of the robust locomotion and DreamWaQ-style policy design used in this repository.
* [`MGDP`](https://github.com/arclab-hku/MGDP) inspired part of the terrain type design (currently not used in the go2w rough task, but will be required for future parkour tasks).
* [`toonasinensis/rsl_rl`](https://github.com/toonasinensis/rsl_rl) inspired part of the modular RSL-RL training architecture and plugin-style organization.

I sincerely thank the authors and contributors of these projects for releasing their code and advancing open research in legged robot learning.

Please cite and respect the licenses of the original projects when using this repository.


## License

The original `legged_gym` and `rsl_rl` components are distributed under the BSD-3-Clause license.

Additional modifications in this fork should follow the same license unless otherwise specified.

---

## Citation

If you use the manager-based Isaac Gym training code, you can cite this repository as:

```bibtex
@misc{legged_manager_2026,
  title        = {Legged Manager: Manager-Based Isaac Gym Training Code for Legged Robot Locomotion},
  author       = {Jingshuo Xie},
  year         = {2026},
  howpublished = {\url{https://github.com/x714543179/legged-manager}}
}
```
