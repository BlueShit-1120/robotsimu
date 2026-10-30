# M1 — 单臂 Reach（重建说明 + 云端冒烟）

对齐 [`rl-pick-place-plan.md`](./rl-pick-place-plan.md) §5 M1 / curriculum C0，决策 D1–D4。

> 本机原结果在 `~/Desktop/spider-robot-sim/src/robotsimu`（分支 `cursor/m1-reach-f7a0`，未 push）。  
> 云端 M2 从 GitHub M0 + 本文规格 **重建** Reach；下列「本机短训」为原 M1 worker 指标，「云端冒烟」为 Grasp 云端附带训练。

---

## 做了什么

- Env：`phase="reach"`，目标 = 盒心上方 hover；奖励 = \(-d\) + 进度塑形 + 成功奖金；`info.reach_success` / `reach_dist` / `reach_target`。
- 夹爪默认 **hold open**（动作维仍为 8）。
- 右臂 MJCF 改为 **position** 伺服（`assets/mjcf/robot_assemble_m0.xml`）。
- 盒 spawn：`(0.70, -0.20)`；短训 hover **0.30 m**。
- 脚本：`scripts/train_reach.py`、`scripts/eval_reach.py`；配置：`configs/ppo_reach.yaml`。

---

## 本机短训结果（原 M1）

| 项 | 值 |
|----|-----|
| timesteps | **100_000** |
| 墙钟 | **~105 s**（~950 FPS，CPU） |
| checkpoint | `runs/reach_m1/ppo_reach.zip` |
| eval | **10/10**，mean dist **≈ 0.039 m**（阈值 5 cm） |

## 云端冒烟（M2 重建）

| 项 | 值 |
|----|-----|
| timesteps | **50_000** |
| checkpoint | `runs/reach_m1/ppo_reach.zip`（云端） |
| eval | **10/10**，mean dist **≈ 0.048 m** |

```bash
python scripts/train_reach.py --timesteps 100000 --out runs/reach_m1 \
  --hover-offset 0.30 --success-dist 0.05 --delta-q-scale 0.06 --device cpu
python scripts/eval_reach.py --checkpoint runs/reach_m1/ppo_reach.zip --episodes 10
```

---

## Gaps → Grasp

- hover 0.30 → grasp 需更低（~0.06–0.08）
- 夹爪 hold-open；无抬升/接触奖励
- 见 [`m2-grasp.md`](./m2-grasp.md)
