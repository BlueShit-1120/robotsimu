# robotsimu — 双平台 RL 抓放（MuJoCo M0–M5 · Isaac Lab M6–M10）

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

文档：仓内 `docs/`；项目侧详细报告见 Agent Store `docs/m*.md`。

---

## Environments / 环境

### MuJoCo（`rs-mujoco` 或 venv）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-mujoco.txt
pip install -e .
# 或：conda env create -f environment-mujoco.yml && conda activate rs-mujoco
```

### Isaac Lab（conda `env_isaacsim`）

见钉扎：Isaac Sim **4.5** · Lab **v2.1.1**（`~/IsaacLab`）· torch **2.7.0+cu128** · SB3 **2.7.1**。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
# 首次 / 重装 RL extras：
cd ~/IsaacLab && ./isaaclab.sh --install all
# 若 torch 被抬升，钉回 2.7.0+cu128 + SB3<2.8（见 docs/m8-isaac-rl-extras.md）
```

---

## Quick start — MuJoCo

```bash
# 冒烟
python scripts/smoke_mujoco.py --steps 100 --seed 0

# M1 Reach
python scripts/train_reach.py --timesteps 100000 --out runs/reach_m1 \
  --hover-offset 0.30 --success-dist 0.05 --delta-q-scale 0.06 --device cpu
python scripts/eval_reach.py --checkpoint runs/reach_m1/ppo_reach.zip --episodes 10

# M2 Grasp（可 --resume Reach ckpt）
python scripts/train_grasp.py --timesteps 200000 --out runs/grasp_m2 \
  --hover-offset 0.06 --lift-height 0.04 --device cpu
python scripts/eval_grasp.py --checkpoint runs/grasp_m2/ppo_grasp.zip --episodes 100

# M3 Place（默认已持物起步）
python scripts/train_place.py --timesteps 100000 --out runs/place_m3 --device cpu
python scripts/eval_place.py --checkpoint runs/place_m3/ppo_place.zip --episodes 100

# M4 无 latch 接触抓取
python scripts/train_grasp.py --timesteps 500000 --out runs/grasp_m4 --no-latch \
  --lift-height 0.03 --delta-q-scale 0.03 --device cpu --seed 17
python scripts/eval_grasp.py --checkpoint runs/grasp_m4/ppo_grasp.zip \
  --episodes 100 --no-latch --lift-height 0.03 --delta-q-scale 0.03

# M5 full-pick（contact_hold + grasp bootstrap；详见 docs/m5-full-pick.md）
python scripts/train_place.py --full-pick --no-latch --contact-hold \
  --bootstrap-grasp runs/grasp_m4/ppo_grasp.zip \
  --timesteps 150000 --out runs/place_m5 --device cpu
python scripts/eval_place.py --checkpoint runs/place_m5/ppo_place.zip \
  --full-pick --no-latch --contact-hold --episodes 100 --stochastic
```

### Code entry / 代码入口

```python
from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

env = PickPlaceMujocoEnv(phase="grasp", hover_offset=0.06, randomize_box=False)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# info: grasp_success, reach_success, reach_dist, phase, …
```

---

## Quick start — Isaac Lab

路径请换成你的 robotsimu 根目录。所有 Isaac 脚本经 `~/IsaacLab/isaaclab.sh -p` 启动。

```bash
conda activate env_isaacsim
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
cd ~/IsaacLab
REPO=/path/to/robotsimu

# M6 场景冒烟
./isaaclab.sh -p $REPO/scripts/smoke_isaac_lab.py --headless

# M7 Reach / Grasp env 冒烟
./isaaclab.sh -p $REPO/scripts/smoke_isaac_reach_grasp.py --headless --phase reach --steps 20
./isaaclab.sh -p $REPO/scripts/smoke_isaac_reach_grasp.py --headless --phase grasp --steps 40 --random-policy

# M9 Reach 短训（闭环冒烟量级）
./isaaclab.sh -p $REPO/scripts/train_isaac_reach.py \
  --headless --timesteps 8192 --num-envs 4 --out runs/isaac_reach_m9 --seed 0

# M10 Grasp 短训（ContactSensor on）
./isaaclab.sh -p $REPO/scripts/train_isaac_grasp.py \
  --headless --timesteps 8192 --num-envs 4 --out runs/isaac_grasp_m10 --seed 0
```

Isaac ManagerBased 工厂：`make_reach_grasp_env_cfg(phase="reach"|"grasp", …)`  
（`src/robot_simu_rl/backends/isaac_lab/reach_grasp_env_cfg.py`）

**本机短训参考（RTX 4070 Ti SUPER）：**

| Run | steps × envs | learn 墙钟 | FPS | ep_rew 趋势 |
|-----|--------------|------------|-----|-------------|
| M9 Reach | 8192 × 4 | ~28 s | ~293 | ≈ −1.0（未收敛） |
| M10 Grasp | 8192 × 4 | ~31 s | ~262 | 2.5 → 4.08 |

`runs/` 默认 gitignore；checkpoint / `train_meta.json` 仅存本地。

---

## Branches / 分支

| Branch | Role |
|--------|------|
| **`develop`** | 日常集成（M0–M10 当前线） |
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
  backends/isaac_lab/           # ManagerBased Reach/Grasp + MDP
  common/                       # 关节名 / 路径
scripts/
  smoke_mujoco.py, train_*.py, eval_*.py
  smoke_isaac_*.py, train_isaac_reach.py, train_isaac_grasp.py
  convert_robot_usd.py, make_proxy_urdf.py
configs/                        # PPO / task YAML
docs/                           # M0–M10 切片说明
runs/                           # 本地训练产物（gitignore）
```

---

## Notes / 注意

- Isaac **无** MuJoCo latch / `contact_hold`；Grasp 仅 PhysX 摩擦 + ContactSensor。  
- Kit `simulation_app.close()` 偶发挂起；训练脚本以 `SystemExit` 强制收尾。  
- 重跑 `./isaaclab.sh --install all` 可能抬升 torch；按 `docs/m8-isaac-rl-extras.md` 钉回。  
- 下一刀候选：加长 Isaac Reach/Grasp、`eval_isaac_*.py`、Place / full-pick on Lab。

---

## License / 归属

机体网格与 URDF/MJCF 位于本仓 `assets/DVT1/`；M0–M10 补丁与 RL 脚手架一并合入本仓。
