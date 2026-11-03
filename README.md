# robotsimu — 双平台 RL 抓放（MuJoCo M0–M5 · Isaac Lab M6–M16）

基于 [robotsimu](https://github.com/BlueShit-1120/robotsimu) DVT1 整机资产，在 **MuJoCo 3** 与 **Isaac Lab** 上实现右臂抓放：代理 Robotiq 夹爪 + 桌面长方体 + 冻结底盘。

| Item / 项 | Value / 值 |
|-----------|------------|
| Action / 动作 | 右臂 7 + 夹爪 1 = **8** DoF |
| Workpiece / 工件 | **5×4×3 cm**，**0.15 kg** |
| Gripper / 夹爪 | 盒/柱 + 球垫指尖代理（无 STL） |
| Base / 底盘 | **冻结** |
| Algo / 算法 | Stable-Baselines3 **PPO** |
| Obs (Isaac/MuJoCo) | **31** 维对齐布局 |

**Default branch / 默认开发分支：** `develop`（跟踪 `origin/develop`）。`main` 为稳定镜像。

**Sim-to-sim：** 观测 / 奖励 / 成功 / 接触 / 控制一页对照 → [`docs/sim-to-sim-table.md`](docs/sim-to-sim-table.md)。

---

## Roadmap / 阶段总览

| Milestone | Backend | Phase / 要点 | Exit / 退出 | Key scripts |
|-----------|---------|--------------|-------------|-------------|
| **M0** | MuJoCo | 资产接入、代理夹爪、冒烟 | 无崩溃 | `smoke_mujoco.py` |
| **M1** | MuJoCo | Reach：盒上 hover，夹爪 hold-open | 距离达标 | `train_reach.py` / `eval_reach.py` |
| **M2** | MuJoCo | Grasp：闭合 + 抬升（默认 latch） | 抬升 ≥70% / ≥100 ep | `train_grasp.py` / `eval_grasp.py` |
| **M3** | MuJoCo | Place：持物搬运 → 释放 | 完整成功率 ≥50% | `train_place.py` / `eval_place.py` |
| **M3b** | MuJoCo | 收紧 tol / 关 latch | 紧公差释放 | 同 Place 脚本 + flags |
| **M4** | MuJoCo | 无 latch 接触抓取（球垫指尖） | 抬升 ≥70% | `train_grasp.py --no-latch` |
| **M5** | MuJoCo | Full-pick：grasp→place + `contact_hold` | place ≥50% stoch | `train_place.py --full-pick` |
| **M6** | Isaac Lab | 代理 URDF→USD + 场景冒烟 | smoke OK | `convert_robot_usd.py` / `smoke_isaac_lab.py` |
| **M7** | Isaac Lab | ManagerBased Reach/Grasp（8/31 对齐） | smoke OK | `smoke_isaac_reach_grasp.py` |
| **M8** | Isaac Lab | Lab RL extras 安装 + torch/SB3 钉扎 | imports OK | `./isaaclab.sh --install all` |
| **M9** | Isaac Lab | Reach SB3 短训闭环 | train meta ok | `train_isaac_reach.py` |
| **M10** | Isaac Lab | Grasp SB3 短训闭环（ContactSensor） | train meta ok | `train_isaac_grasp.py` |
| **M12** | Isaac Lab | Reach 控制重调（锁存 Δq） | ≥7/10 可比 M1 | `train_isaac_reach.py` |
| **M14–M15** | Isaac Lab | Grasp tip 接触栈 / ContactSensor 持物 | 力可读；hold 指标 | `train_isaac_grasp.py` |
| **M16** | Isaac Lab | Place：hold-assist + C3 释放短训 | place ≥50% | `train_isaac_place.py` / `eval_isaac_place.py` |

文档：仓内 `docs/`（含 [sim-to-sim 对照表](docs/sim-to-sim-table.md)）；项目侧详细报告见 Agent Store `docs/m*.md`。

---

## Environments / 环境激活

在仓库根目录操作（下文记为 `REPO`，本机常见路径：`/home/xiaowen/Desktop/spider-robot-sim/src/robotsimu`）。

### MuJoCo

```bash
cd /path/to/robotsimu
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-mujoco.txt
pip install -e .
# 或：conda env create -f environment-mujoco.yml && conda activate rs-mujoco
```

### Isaac Lab（conda `env_isaacsim`）

钉扎：Isaac Sim **4.5** · Lab **v2.1.1**（`~/IsaacLab`）· torch **2.7.0+cu128** · SB3 **2.7.1**。

**每次跑 Isaac 脚本前必做：**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm PYTHONUNBUFFERED=1
cd ~/IsaacLab
REPO=/path/to/robotsimu   # 换成你的 robotsimu 根目录
```

首次 / 重装 RL extras：`./isaaclab.sh --install all`（若 torch 被抬升，钉回 2.7.0+cu128 + SB3&lt;2.8，见 `docs/m8-isaac-rl-extras.md`）。

所有 Isaac 命令经 `./isaaclab.sh -p … --headless` 启动；相对 `--out` / `--checkpoint` 会落到 `$REPO/runs/…`（`runs/` gitignore，仅本地）。

---

## 本地命令速查 — MuJoCo

在已激活的 MuJoCo 环境中、于 `REPO` 根目录执行。

### 冒烟

```bash
python scripts/smoke_mujoco.py --steps 100 --seed 0
```

### Reach（M1）

```bash
python scripts/train_reach.py --timesteps 100000 --out runs/reach_m1 \
  --hover-offset 0.30 --success-dist 0.05 --delta-q-scale 0.06 --device cpu --seed 0
# checkpoint: runs/reach_m1/ppo_reach.zip

python scripts/eval_reach.py --checkpoint runs/reach_m1/ppo_reach.zip \
  --episodes 10 --hover-offset 0.30 --success-dist 0.05 --delta-q-scale 0.06 \
  --out runs/reach_m1/eval_summary.json
```

### Grasp（M2 latch / M4 无 latch）

```bash
# M2：默认 latch；可 --resume runs/reach_m1/ppo_reach.zip
python scripts/train_grasp.py --timesteps 200000 --out runs/grasp_m2 \
  --hover-offset 0.06 --lift-height 0.03 --device cpu --seed 0
# checkpoint: runs/grasp_m2/ppo_grasp.zip

python scripts/eval_grasp.py --checkpoint runs/grasp_m2/ppo_grasp.zip \
  --episodes 100 --hover-offset 0.06 --lift-height 0.03 \
  --out runs/grasp_m2/eval_summary.json

# M4：无 latch 接触抓取
python scripts/train_grasp.py --timesteps 500000 --out runs/grasp_m4 --no-latch \
  --lift-height 0.03 --delta-q-scale 0.03 --device cpu --seed 17
python scripts/eval_grasp.py --checkpoint runs/grasp_m4/ppo_grasp.zip \
  --episodes 100 --no-latch --lift-height 0.03 --delta-q-scale 0.03 \
  --out runs/grasp_m4/eval_summary.json
```

### Place（M3 短训 / M5 full-pick）

```bash
# M3：默认已持物起步（非 --full-pick）；示例对齐 docs/m3-place.md
python scripts/train_place.py --timesteps 100000 --out runs/place_m3 \
  --place-tol-xy 0.12 --place-tol-z 0.04 --place-stable-steps 2 --device cpu --seed 0
# checkpoint: runs/place_m3/ppo_place.zip

python scripts/eval_place.py --checkpoint runs/place_m3/ppo_place.zip \
  --episodes 100 --place-tol-xy 0.12 --place-tol-z 0.04 --place-stable-steps 2 \
  --out runs/place_m3/eval_summary.json

# M5：full-pick + contact_hold（详见 docs/m5-full-pick.md）
python scripts/train_place.py --full-pick --no-latch --contact-hold \
  --bootstrap-grasp runs/grasp_m4/ppo_grasp.zip \
  --timesteps 150000 --out runs/place_m5 --device cpu
python scripts/eval_place.py --checkpoint runs/place_m5/ppo_place.zip \
  --full-pick --no-latch --contact-hold --episodes 100 --stochastic \
  --out runs/place_m5/eval_summary.json
```

### 代码入口

```python
from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

env = PickPlaceMujocoEnv(phase="grasp", hover_offset=0.06, randomize_box=False)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# info: grasp_success, reach_success, reach_dist, phase, …
```

---

## 本地命令速查 — Isaac Lab

先完成上文 **Isaac 环境激活**（`env_isaacsim` + `OMNI_KIT_ACCEPT_EULA` + `PYTHONNOUSERSITE=1`），在 `~/IsaacLab` 下执行。工厂：`make_reach_grasp_env_cfg(phase="reach"|"grasp"|"place", …)`。

### 冒烟（Reach / Grasp / Place）

```bash
./isaaclab.sh -p $REPO/scripts/smoke_isaac_lab.py --headless

./isaaclab.sh -p $REPO/scripts/smoke_isaac_reach_grasp.py --headless \
  --phase reach --steps 20
./isaaclab.sh -p $REPO/scripts/smoke_isaac_reach_grasp.py --headless \
  --phase grasp --steps 40 --random-policy
./isaaclab.sh -p $REPO/scripts/smoke_isaac_reach_grasp.py --headless \
  --phase place --steps 40
```

### Reach（M12 推荐）

```bash
./isaaclab.sh -p $REPO/scripts/train_isaac_reach.py --headless \
  --timesteps 200000 --num-envs 4 --out runs/isaac_reach_m12 --seed 0 \
  --delta-q-scale 0.06 --n-steps 512 --hover-offset 0.30 --success-dist 0.05
# checkpoint: $REPO/runs/isaac_reach_m12/ppo_isaac_reach.zip

./isaaclab.sh -p $REPO/scripts/eval_isaac_reach.py --headless \
  --checkpoint runs/isaac_reach_m12/ppo_isaac_reach.zip \
  --episodes 10 --num-envs 1 --delta-q-scale 0.06 \
  --hover-offset 0.30 --success-dist 0.05 --seed 0 \
  --out runs/isaac_reach_m12/eval_summary.json
```

### Grasp（M14/M15 接触栈）

```bash
./isaaclab.sh -p $REPO/scripts/train_isaac_grasp.py --headless \
  --timesteps 200000 --num-envs 4 --out runs/isaac_grasp_m15 --seed 2 \
  --ent-coef 0.03 --delta-q-scale 0.03 --hover-offset 0.06 \
  --lift-height 0.03 --max-episode-steps 160
# 可选续训：--resume runs/isaac_grasp_m14b/ppo_isaac_grasp.zip
# checkpoint: $REPO/runs/isaac_grasp_m15/ppo_isaac_grasp.zip

./isaaclab.sh -p $REPO/scripts/eval_isaac_grasp.py --headless \
  --checkpoint runs/isaac_grasp_m15/ppo_isaac_grasp.zip \
  --episodes 20 --num-envs 1 --delta-q-scale 0.03 \
  --hover-offset 0.06 --lift-height 0.03 --max-episode-steps 160 \
  --out runs/isaac_grasp_m15/eval_summary.json
```

### Place（M16 C3 释放短训）

默认 `place_start_holding`（闭爪 + hold-assist）；可用 `--no-place-start-holding` 关掉。

```bash
./isaaclab.sh -p $REPO/scripts/train_isaac_place.py --headless \
  --timesteps 50000 --num-envs 4 --out runs/isaac_place_m16 --seed 0 \
  --delta-q-scale 0.03 --lift-height 0.03 \
  --place-tol-xy 0.12 --place-tol-z 0.04 --place-stable-steps 2 \
  --max-episode-steps 160
# checkpoint: $REPO/runs/isaac_place_m16/ppo_isaac_place.zip

./isaaclab.sh -p $REPO/scripts/eval_isaac_place.py --headless \
  --checkpoint runs/isaac_place_m16/ppo_isaac_place.zip \
  --episodes 20 --num-envs 1 --delta-q-scale 0.03 --lift-height 0.03 \
  --place-tol-xy 0.12 --place-tol-z 0.04 --place-stable-steps 2 \
  --out runs/isaac_place_m16/eval_summary.json
```

**本机参考（RTX 4070 Ti SUPER）：** M12 Reach ~200k ≈216 FPS；M15 Grasp ~200k ≈160 FPS；M16 Place 50k ≈153 FPS。闭环冒烟量级可用 `--timesteps 8192`。

---

## Branches / 分支

| Branch | Role |
|--------|------|
| **`develop`** | 日常集成（M0–M16 当前线） |
| `main` | 稳定镜像 |
| `cursor/m5-full-pick` 等 | 历史切片分支（可选对照） |

```bash
git fetch && git checkout develop && git pull origin develop
```

---

## Layout / 目录

```text
assets/mjcf/                    # MuJoCo 代理机器人 + 抓放场景
assets/isaac/                   # 代理 URDF；USD 由 convert 生成（gitignore）
assets/DVT1/                    # 机体网格 / URDF 源
src/robot_simu_rl/
  backends/mujoco/              # Gymnasium PickPlaceMujocoEnv
  backends/isaac_lab/           # ManagerBased Reach/Grasp/Place + MDP
  common/                       # 关节名 / 路径
scripts/
  smoke_mujoco.py, train_*.py, eval_*.py
  smoke_isaac_*.py, train_isaac_{reach,grasp,place}.py, eval_isaac_*.py
  convert_robot_usd.py, make_proxy_urdf.py
configs/                        # PPO / task YAML
docs/                           # 切片说明 + sim-to-sim-table.md
runs/                           # 本地训练产物（gitignore）
```

---

## Notes / 注意

- Isaac Grasp **无** MuJoCo latch / `contact_hold`（仅 PhysX 摩擦 + ContactSensor）；Place 短训用 kinematic **hold-assist** 作 latch 替身。详见 [`docs/sim-to-sim-table.md`](docs/sim-to-sim-table.md)。  
- Kit `simulation_app.close()` 偶发挂起；训练脚本以 `SystemExit` 强制收尾。  
- 重跑 `./isaaclab.sh --install all` 可能抬升 torch；按 `docs/m8-isaac-rl-extras.md` 钉回。  
- 下一刀候选：Isaac full-pick（Grasp→Place）、Grasp 稳态终态抬升、收紧 Place tol。

---

## License / 归属

机体网格与 URDF/MJCF 位于本仓 `assets/DVT1/`；M0–M16 补丁与 RL 脚手架一并合入本仓。
