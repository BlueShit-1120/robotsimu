# M0 — MuJoCo 资产接入与代理几何

对齐方案 [`rl-pick-place-plan.md`](./rl-pick-place-plan.md) §5 M0 与决策 D1–D4。

## 已完成

- In-repo assets `assets/DVT1/` (was robotsimu@6342d4c vendor pin)
- 代理夹爪/FT 几何 → `assets/mjcf/robot_assemble_m0.xml`
- 桌面 + 5×4×3 cm / 0.15 kg 工件 → `assets/mjcf/pick_place_scene.xml`
- 冻 `floating_base`；env 锁轮/左臂/头/躯干常值
- Gymnasium `PickPlaceMujocoEnv`（8 维动作）
- `scripts/smoke_mujoco.py` 冒烟

## 关节名映射（节选）

| 逻辑 | URDF | MJCF |
|------|------|------|
| arm_r_j1…7 | `right_arm_joint1…7` | 同左 |
| grip_r | `right_robotiq_85_left_knuckle_joint` | `right_gripper_left_finger_joint` |

## 非目标（本切片）

Isaac Lab 任务、PPO 训练、高保真夹爪 STL、CNC 场景恢复。
