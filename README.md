# robotsimu — RL 抓放（M0/M1 MuJoCo）

基于 [robotsimu](https://github.com/BlueShit-1120/robotsimu) DVT1 整机资产，在 **MuJoCo 3** 上实现右臂抓放课程：**M0** 代理夹爪场景 → **M1** Reach（EE 到盒上方预抓取位）。

| 项 | 值 |
|----|-----|
| 动作 | 右臂 7 + 夹爪 1 = **8** DoF（Reach 默认夹爪保持张开） |
| 工件 | **5×4×3 cm**，**0.15 kg**，μ≈**0.6 / 0.5** |
| 夹爪 | 盒/柱 **代理几何**（无 STL） |
| 底盘 | **冻结**（无 `floating_base`） |
| 右臂控制 | MJCF **position** 伺服（`kp/kv` + `forcerange`） |
| 资产 | 本仓 `assets/DVT1/` + `assets/mjcf/` |

Isaac Lab / Isaac Sim 不阻塞本切片（本机安装见项目决策 D3）。

---

## 快速开始 / Quick start

### 环境 `rs-mujoco`

```bash
# Conda（推荐）
conda env create -f environment-mujoco.yml
conda activate rs-mujoco

# 或 pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-mujoco.txt
pip install -e .
# RL extras（若未装）:
pip install 'stable-baselines3>=2.3' torch tensorboard
# CUDA torch（可选）:
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 冒烟 / Smoke

```bash
python3 scripts/check_assets.py
python3 scripts/smoke_mujoco.py --steps 100 --seed 0
python3 scripts/smoke_mujoco.py --policy zero
python3 scripts/smoke_mujoco.py --policy open_close
# 可选 viewer
python3 scripts/smoke_keyboard.py
```

期望：`[smoke] OK — completed without crash`。

### M1 Reach 训练 / 评估

```bash
conda activate rs-mujoco
cd /path/to/robotsimu

# 短训（约 1–3 分钟 @ ~900 FPS CPU）
python scripts/train_reach.py \
  --timesteps 100000 \
  --out runs/reach_m1 \
  --hover-offset 0.30 \
  --success-dist 0.05 \
  --delta-q-scale 0.06 \
  --device cpu

# 评估
python scripts/eval_reach.py \
  --checkpoint runs/reach_m1/ppo_reach.zip \
  --episodes 10 \
  --hover-offset 0.30 \
  --success-dist 0.05

# 续训更长（例如压到 3 cm / 更低 hover）
python scripts/train_reach.py \
  --resume runs/reach_m1/ppo_reach.zip \
  --timesteps 300000 \
  --out runs/reach_m1_long \
  --hover-offset 0.15 \
  --success-dist 0.03
```

checkpoint 默认：`runs/reach_m1/ppo_reach.zip`；摘要：`runs/reach_m1/eval_summary.json`。

### 代码入口

```python
from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv
from robot_simu_rl.tasks.pick_place import ReachConfig

env = PickPlaceMujocoEnv(
    phase="reach",
    randomize_box=False,
    hold_gripper_open=True,
    reach_cfg=ReachConfig(hover_offset_m=0.30, success_dist_m=0.05),
)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# info: reach_dist, reach_target, reach_success, phase
```

---

## 目录结构

```text
assets/
  DVT1/                    # URDF / meshes / upstream MJCF
  mjcf/
    robot_assemble_m0.xml  # 代理夹爪 + 冻底盘 + 右臂 position 伺服
    pick_place_scene.xml   # 桌面 + 工件
configs/
  task_pick_place.yaml
  ppo_reach.yaml
src/robot_simu_rl/
  common/                  # 关节名映射 URDF↔MJCF
  backends/mujoco/         # Gymnasium env
  tasks/pick_place/        # Reach 奖励 / 成功判据
scripts/
  smoke_mujoco.py
  train_reach.py
  eval_reach.py
runs/                      # 本地训练输出（默认 gitignore）
```

重新生成代理 MJCF：`python3 scripts/regen_m0_mjcf.py`（若覆盖右臂 actuator，需再改回 position 伺服）。

---

## M1 已知缺口 / Gaps toward grasp（M2）

- 当前 hover **0.30 m**（便于短训）；真抓取需降到 ~**0.08 m** 并加长训。
- 夹爪在 Reach 阶段 **hold open**；M2 需释放夹爪动作 + 接触/抬升奖励。
- 盒位默认固定（C0）；M2+ 再开 `randomize_box` / 域随机。
- 成功阈值短训用 **5 cm**；方案退出目标 **2–3 cm** 需续训。
- 放置目标与完整 grasp/place 奖励尚未接（M2–M3）。
- Isaac Lab 适配未实现（D3）。

---

## 许可与归属

机体网格与 URDF/MJCF 位于本仓 `assets/DVT1/`；M0/M1 补丁 MJCF 与 RL 脚手架一并合入本仓。
