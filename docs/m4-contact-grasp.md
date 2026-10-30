# M4 — Contact grasp without latch（云端结果）

Follow-up to [`m3b-tighten.md`](./m3b-tighten.md) / [`m2-grasp.md`](./m2-grasp.md)。目标：关掉 curriculum latch，靠代理指**真实接触**抬起工件。  
Agent：`bc-ca6d3804-1f54-51b6-ab0f-50fd5e1ee981` · 分支 `cursor/m4-contact-grasp-e981`（基于 M3b / M2 grasp）。

---

## 做了什么

- **指垫**：右指尖 collision 改为 sticky **sphere**（r=0.018），`friction≈3.5`，`condim=6`；其余右爪 knuckle/finger/base collision 关闭，避免顶飞盒子。
- **接触掩码**：指尖 `contype=2`，工件 `conaffinity=3` → 指尖↔盒接触、**指尖↔桌面不碰**（解决闭合下潜穿桌）。
- **求解**：`noslip_iterations=20`，`impratio=20`，`cone=elliptic`；夹爪 `kp=70` + `ctrlrange`/`forcerange` 限幅；右臂 position kp 上调。
- **动作/课程**：grasp home 按「闭合后 tip mid 仍在盒心」优化；`delta_q_scale=0.03`；奖励改为 tip-mid→盒 + 双指接触 shaping；PPO `log_std_init=-1`。
- **训练**：`--no-latch`，500k steps（CPU），checkpoint `runs/grasp_m4/ppo_grasp.zip`。
- 产物：`media/m4-contact-grasp-cloud.tar.gz`。

---

## 对比指标（主表）

| 配置 | latch | steps | Δq | eval lift success | mean box_z | mean ee–box | 备注 |
|------|-------|-------|----|-------------------|------------|-------------|------|
| M2 grasp baseline | **on** | 250k | 0.05 | **100/100** | lift ok | — | 运动学吸附助教 |
| M3b grasp no-latch | **off** | 150k | 0.05 | **0/100** | 0.765 | 0.17 | 旧代理指，未抬起 |
| **M4 contact（本切片）** | **off** | **500k** | **0.03** | **100/100** | **0.807** | 0.125 | 真接触；exit ≥70% **达标** |
| M4 脚本 oracle（对照） | off | — | 0.03 | 20/20 | ~0.80 | — | tip-mid 伺服 + 闭合抬升 |

抬升阈值：`lift_height=0.03` m（与 M2 口径一致）。

---

## 短训 / 评估命令

```bash
pip install -r requirements-mujoco.txt && pip install -e .

python scripts/train_grasp.py --timesteps 500000 --out runs/grasp_m4 \
  --no-latch --hover-offset 0.06 --lift-height 0.03 --delta-q-scale 0.03 \
  --max-episode-steps 160 --device cpu --seed 17 --ent-coef 0.01

python scripts/eval_grasp.py --checkpoint runs/grasp_m4/ppo_grasp.zip \
  --episodes 100 --no-latch --hover-offset 0.06 --lift-height 0.03 \
  --delta-q-scale 0.03 --max-episode-steps 160
```

墙钟：约 **691 s**（~724 FPS）· `runs/grasp_m4/train_meta.json` / `eval_summary.json`。

---

## 结论

1. **无 latch 接触抓取已打通**：M3b 的 0% → M4 **100/100** 抬升成功（mean box_z≈0.807，抬约 4.2 cm）。
2. 关键杠杆是 **指尖球体垫 + 高摩擦/noslip + 指–桌不碰撞 + 闭合高度 grasp home + 更小 Δq**，而不是加回 latch。
3. 旧 M3b no-latch ckpt 在新物理上仍为 0%（需按本切片重训）。

---

## Gaps

- Place / full-pick 在无 latch 接触抓取上的衔接未跑（下一刀可 `--full-pick --no-latch` resume M4）。
- 代理球垫偏乐观于真 Robotiq 指面；sim-to-sim / 真机前需再收紧几何。
- C4 盒位域随机未开；策略对固定 spawn 过拟合风险仍在。
