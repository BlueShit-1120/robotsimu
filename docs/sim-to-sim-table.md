# MuJoCo ↔ Isaac Lab 对照表（一页）

> 日期：2026-11-03 · RS-31 · Agent：`bc-5165de52-7f25-584a-a95b-24ae2a2c8bfb`  
> 仓：`robotsimu` · 默认分支 **`develop`**  
> MuJoCo：`PickPlaceMujocoEnv` · Isaac：`make_reach_grasp_env_cfg` / ManagerBasedRLEnv  
> 阶段：Reach (M1/M12) · Grasp (M4/M14–M15) · Place (M3/M16)

**原则：** 任务语义一套、后端两套。本表只标**对齐点与已知 gap**，消除 sim-to-sim 歧义。

---

## 1. 接口总览

| 项 | MuJoCo | Isaac Lab | 状态 |
|----|--------|-----------|------|
| 动作 | 8：Δq×7 + grip∈[0,1] | 同；Lab Box 名义无界，项内 clip | **对齐** |
| 观测 | 31：`qn(7)+dqn(7)+grip(1)+ee(7)+box(7)+goal_xy(2)` | 同布局（`mujoco_style_obs`） | **对齐** |
| EE / box 系 | 世界系 | 减 `env_origins`（多 env） | 单 env **等价** |
| quat | MuJoCo `xquat` wxyz | PhysX body quat wxyz | **对齐** |
| 控制周期 | `frame_skip`×`timestep` | `decimation`×`sim.dt`（默认 0.02 s） | 默认 **对齐量级** |
| 算法入口 | Gymnasium 单 env + SB3 | ManagerBased 向量 + `Sb3VecEnvWrapper` | 接口不同、策略维同 |
| 底盘 | 冻结 | USD `fix_base` + 高刚度 | **对齐意图** |

---

## 2. 控制（Δq / 夹爪）

| 项 | MuJoCo | Isaac Lab | Gap |
|----|--------|-----------|-----|
| 臂指令 | 每控步 `q_des = q + scale·clip(a)` | **锁存积分** `q_des += scale·a`（M12；防 PhysX 漂移追赶） | 语义近；Isaac ≠ 每步重读 `q` |
| `delta_q_scale` | Reach **0.06**；Grasp/Place 常 **0.03** | 同约定（脚本默认） | **对齐** |
| 臂伺服 | MuJoCo 位置执行器（偏刚） | PD：`effort≈1e5`、`kp≈2000`（M12） | 跟踪比不同 |
| 夹爪 | cmd∈[0,1]→ knuckle 开合；MJCF equality mimic | `AbsoluteGripperAction` + 脚本写 mimic 组 | Mimic 靠 PhysX 跟随 |
| Reach home | near-home | `arm_home="near"`（phase=reach） | **对齐（M12）** |
| Grasp/Place home | grasp-home | `arm_home="grasp"` | **对齐** |

---

## 3. 观测（31 维，逐段）

| 切片 | 维 | 定义（两端相同） | 备注 |
|------|----|------------------|------|
| `qn` | 7 | 右臂关节角归一化到关节限位中点 | |
| `dqn` | 7 | 关节速度 / π | |
| `grip` | 1 | knuckle 角归一化到 `[lo,hi]` | |
| `ee` | 7 | EE pos(3)+quat(4) | Isaac 减 env origin |
| `box` | 7 | 工件 pos(3)+quat(4) | 同上 |
| `goal_xy` | 2 | Place 目标 **(0.70, −0.12)** | Reach/Grasp 也写入（parity） |

无额外 contact / latch 标志进 obs（两端皆无）。

---

## 4. 奖励（塑形意图）

Lab `RewardManager` 乘 `step_dt`；配置侧用 **`weight = 1/step_dt`**（M12）使**标量与 MuJoCo 逐步奖励同量级**。

| Phase | MuJoCo | Isaac | Gap |
|-------|--------|-------|-----|
| **Reach** | −d + 2·progress；成功 +5 / 首次 +10 | 同塑形 | **对齐** |
| **Grasp** | tip-mid→盒、近距闭合、接触、抬升；无 latch 时靠 geom 接触 | 同塑形 + 力/几何 pinch；sustained `hold_steps` | 权重经 M14/M15 加严；**非逐项拷贝** |
| **Place** | C3 开爪课程 + carry；成功 +200 / 首次 +300 | 同塑形（`place_reward`） | 短训路径 **对齐** |

---

## 5. 成功判据

| Phase | MuJoCo | Isaac | Gap |
|-------|--------|-------|-----|
| **Reach** | ‖EE − hover‖ &lt; `success_dist`（默认 0.05） | 同 | **对齐** |
| **Grasp（latch）** | latched ∧ lift ∧ grip&gt;0.25 | **无 latch 路径** | Isaac 不做 proximity latch |
| **Grasp（接触）** | tip 接触（或近）∧ grip ∧ lift | **sustained** pinch∧closed∧lift ≥ `hold_steps`（默认 15≈0.3 s） | Isaac **更严**（M15） |
| **Place** | XY&lt;tol ∧ \|z−z0\|&lt;tol_z ∧ ¬latched ∧ grip&lt;0.35 ∧ acquired，连续 `place_stable_steps` | 同；`¬hold_assist` 代替 ¬latched | **对齐** |
| 早停 | `early_terminate_success` | 同（TerminationManager） | **对齐** |

Place 默认 tol：`xy=0.12`、`z=0.04`、`stable_steps=2`（M3/M16）。

---

## 6. 接触 / 持物

| 项 | MuJoCo | Isaac Lab | Gap |
|----|--------|-----------|-----|
| 指尖接触检测 | `data.contact` geom 对（tip↔box） | `ContactSensor` net+matrix；工件须 `activate_contact_sensors=True`（M15） | API 不同；修好后可读力 |
| 指垫摩擦 | 球垫 μ≈3.5、noslip | tip 材质 μ≈3.5 + 工件 μ≈3.5；PhysX friction 迭代↑（M14） | 近似；非 MJ 球垫几何 |
| tip↔桌过滤 | MJCF 接触过滤 | FilteredPairs：tip↔Table/Ground | **对齐意图** |
| **latch** | 近距运动学吸附（M2/M3 短训） | Grasp **无**；Place **hold-assist**（tip-mid 贴盒，开爪释放） | Place 替身；Grasp 纯接触 |
| **contact_hold** | 双指接触后焊接持物（M5） | **无**等价焊接 | **最大持物 gap** |
| holding 语义 | latch 或 cmd/几何 | Grasp：力∪几何 pinch；Place：hold-assist 标志 | 指标名同、实现异 |

---

## 7. 短训结果快照（可比口径）

| Phase | MuJoCo | Isaac | 备注 |
|-------|--------|-------|------|
| Reach | M1：10/10 @100k | M12：10/10 @200k，mean dist≈0.042 | 控制修好后可比 |
| Grasp | M4 no-latch：抬升≥70%@500k | M15：sustained 5/20；any_hold 90%；峰 tip≈411 N | 成功定义更严；终态抬升仍弱 |
| Place | M3：100/100 @100k（start holding） | M16：11/20=**55%** @50k（start holding） | 均 ≥50% exit；MuJoCo 更易 |

---

## 8. 仍未对齐（勿混读指标）

1. Isaac Grasp **无** `contact_hold` / latch → 不能直接拿 M2 latch 成功率对比 M15。  
2. Isaac Grasp 成功 = **持续** pinch+lift；MuJoCo M4 多为瞬时接触+抬升。  
3. Place hold-assist ≠ 摩擦托举；full-pick（Grasp→Place）Isaac **未**闭环。  
4. 奖励逐步值对齐靠 `1/dt`；子项系数 Grasp 侧已偏离早期 M7 草稿。  
5. `runs/` 本地 gitignore；对照实验需自备 checkpoint。

---

## 路径

| 端 | 入口 |
|----|------|
| MuJoCo env | `src/robot_simu_rl/backends/mujoco/env.py` |
| Isaac cfg / MDP | `.../isaac_lab/reach_grasp_env_cfg.py` · `mdp/{observations,actions,rewards,events}.py` |
| 报告 | `m1`/`m3`/`m4`/`m12`/`m15`/`m16-isaac-place.md` |
