# robotsimu — RL 抓放（M0 MuJoCo）

基于 [robotsimu](https://github.com/BlueShit-1120/robotsimu) DVT1 整机资产，在 **MuJoCo 3** 上实现右臂抓放第一切片（M0）：代理 Robotiq 夹爪几何 + 桌面长方体工件 + 冻结底盘。

| 项 | 值 |
|----|-----|
| 动作 | 右臂 7 + 夹爪 1 = **8** DoF |
| 工件 | **5×4×3 cm**，**0.15 kg**，μ≈**0.6 / 0.5** |
| 夹爪 | 盒/柱 **代理几何**（无 STL） |
| 底盘 | **冻结**（无 `floating_base`） |
| 资产 | 本仓 `assets/DVT1/`（原 robotsimu 资产树；旧 stub 见 `README.assets.md`） |

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
```

### 冒烟 / Smoke

```bash
# 随机策略步进（无显示，应无崩溃）
python3 scripts/smoke_mujoco.py --steps 100 --seed 0

# 零动作 / 开合夹爪
python3 scripts/smoke_mujoco.py --policy zero
python3 scripts/smoke_mujoco.py --policy open_close

# 可选：被动 viewer（需本机显示）
python3 scripts/smoke_keyboard.py
```

期望输出末尾：`[smoke] OK — completed without crash`。

### 代码入口

```python
from robot_simu_rl.backends.mujoco import PickPlaceMujocoEnv

env = PickPlaceMujocoEnv()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

---

## 目录结构

```text
assets/
  DVT1/                    # URDF / meshes / upstream MJCF（本仓原生）
  mjcf/
    robot_assemble_m0.xml  # 代理夹爪 + 冻底盘
    pick_place_scene.xml   # 桌面 + 工件
  scenes/                  # 既有场景 YAML
src/robot_simu_rl/
  common/                  # 关节名映射 URDF↔MJCF
  backends/mujoco/         # Gymnasium env
  tasks/pick_place/
scripts/smoke_mujoco.py
configs/task_pick_place.yaml
```

重新生成代理 MJCF：`python3 scripts/regen_m0_mjcf.py`。

---

## M0 已知缺口 / Known gaps

- 奖励仅为 EE–物体距离塑形；完整 grasp/place 成功判据与 curriculum 在 M1–M3。
- 冻结关节通过每步写回 `qpos`（非 equality weld）；轮/左臂不暴露。
- 代理指外形与真 2F-85 有偏差；接触摩擦为起步值。
- 机器人碰撞 geom 仍为满 STL，仿真偏慢；`noslip_iterations=2`（上游常用 20）。
- Isaac Lab 适配未实现（见 D3）。
- 键盘脚本仅为 viewer 冒烟，无完整键位遥操作。

---

## 许可与归属

机体网格与 URDF/MJCF 位于本仓 `assets/DVT1/`；M0 补丁 MJCF 与 RL 脚手架一并合入本仓。
