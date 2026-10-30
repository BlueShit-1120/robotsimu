# robotsimu — RL 抓放（M0–M4 MuJoCo）

基于 [robotsimu](https://github.com/BlueShit-1120/robotsimu) DVT1 整机资产，在 **MuJoCo 3** 上实现右臂抓放：代理 Robotiq 夹爪 + 桌面长方体 + 冻结底盘；课程 **Reach → Grasp → Place**（M4：无 latch 接触抓取）。

| 项 | 值 |
|----|-----|
| 动作 | 右臂 7 + 夹爪 1 = **8** DoF |
| 工件 | **5×4×3 cm**，**0.15 kg**（指面摩擦已加强，见 M4） |
| 夹爪 | 盒/柱 + **球垫指尖**代理几何（无 STL） |
| 底盘 | **冻结** |
| 算法 | Stable-Baselines3 **PPO** |

---

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-mujoco.txt
pip install -e .

# 冒烟
python scripts/smoke_mujoco.py --steps 100 --seed 0

# M1 Reach 短训 / 评估
python scripts/train_reach.py --timesteps 100000 --out runs/reach_m1 \
  --hover-offset 0.30 --success-dist 0.05 --delta-q-scale 0.06 --device cpu
python scripts/eval_reach.py --checkpoint runs/reach_m1/ppo_reach.zip --episodes 10

# M2 Grasp 短训 / 评估（可 --resume Reach ckpt）
python scripts/train_grasp.py --timesteps 200000 --out runs/grasp_m2 \
  --hover-offset 0.06 --lift-height 0.04 --device cpu
python scripts/eval_grasp.py --checkpoint runs/grasp_m2/ppo_grasp.zip --episodes 100

# M3 Place 短训 / 评估（默认已持物起步）
python scripts/train_place.py --timesteps 100000 --out runs/place_m3 --device cpu
python scripts/eval_place.py --checkpoint runs/place_m3/ppo_place.zip --episodes 100

# M4 无 latch 接触抓取
python scripts/train_grasp.py --timesteps 500000 --out runs/grasp_m4 --no-latch \
  --lift-height 0.03 --delta-q-scale 0.03 --device cpu --seed 17
python scripts/eval_grasp.py --checkpoint runs/grasp_m4/ppo_grasp.zip \
  --episodes 100 --no-latch --lift-height 0.03 --delta-q-scale 0.03
```

Conda：`conda env create -f environment-mujoco.yml && conda activate rs-mujoco`。

### 代码入口

```python
from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

env = PickPlaceMujocoEnv(phase="grasp", hover_offset=0.06, randomize_box=False)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# info: grasp_success, reach_success, reach_dist, phase, …
```

---

## 阶段

| 阶段 | `phase=` | 要点 | 退出 |
|------|----------|------|------|
| M0 | `m0_smoke` | 加载资产 / 冒烟 | 无崩溃 |
| M1 | `reach` | 盒上 hover；夹爪 hold-open；右臂 **position** 伺服 | 距离达标 |
| M2 | `grasp` | 闭合 + 抬升（C1）；默认 latch 助教 | 抬升成功率 ≥ 70%（≥100 ep） |
| M3 | `place` | 持物搬运 → 松开放置（C2–C3） | 完整成功率 ≥ 50%（≥100 ep） |
| M4 | `grasp` + `--no-latch` | 球垫指尖真接触抓取 | 抬升成功率 ≥ 70%（≥100 ep） |

---

## 目录

```text
assets/mjcf/robot_assemble_m0.xml   # 代理夹爪 + 右臂 position 伺服
assets/mjcf/pick_place_scene.xml
src/robot_simu_rl/backends/mujoco/  # Gymnasium env（phase）
scripts/{smoke_mujoco,train_reach,eval_reach,train_grasp,eval_grasp}.py
configs/{task_pick_place,ppo_reach,ppo_grasp}.yaml
runs/{reach_m1,grasp_m2}/           # checkpoints（训练后）
```

---

## 许可与归属

机体网格与 URDF/MJCF 位于本仓 `assets/DVT1/`；M0–M2 补丁与 RL 脚手架一并合入本仓。
