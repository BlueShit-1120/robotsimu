# M5 — Full-pick without proximity latch（云端结果）

Follow-up to [`m4-contact-grasp.md`](./m4-contact-grasp.md) / [`m3b-tighten.md`](./m3b-tighten.md)。目标：`--full-pick --no-latch`，在 M4 接触指垫上跑通 grasp→place，争取非零 place success。  
Agent：`bc-651c3a9d-4657-5f79-acaa-682e5163f2bc` · 分支 `cursor/m5-full-pick-f2bc`（基于 M4 contact grasp）。

---

## 做了什么

- **基线**：解压 `m4-contact-grasp-cloud.tar.gz`（指尖 sticky sphere + noslip；`runs/grasp_m4/ppo_grasp.zip`）。
- **发现**：M4「抬升成功」多为瞬时弹起，代理球垫**无法靠摩擦持续搬运**；纯 no-hold fine-tune 会冲掉抓取。
- **M5 杠杆**（仍 `--no-latch`，关闭 proximity latch）：
  1. **`contact_hold`**：仅在双指尖↔盒接触 + 闭合时焊接持物；远目标处忽略误张爪，**靠近放置目标才允许释放**。
  2. **`GraspBootstrapWrapper`**：冻结 M4 grasp 直到接触持物，再交给 place PPO（级联；place 网在 buffer 里也会模仿抓取段）。
  3. Full-pick 复位用 M4 grasp home；place Stage-A tip-mid 塑形；place success 要求本局曾持物（避免出生点已在 tol 内的假成功）。
- **训练**：`--full-pick --no-latch --contact-hold --bootstrap-grasp runs/grasp_m4/ppo_grasp.zip`，`place_tol_xy=0.10`，Δq=0.03；约 **150k + 100k + 80k** CPU 步（末段降熵）。
- 产物：`media/m5-full-pick-cloud.tar.gz`。

---

## 对比指标（主表）

| 配置 | latch | hold | start | tol_xy | steps | eval place | mean xy | 备注 |
|------|-------|------|-------|--------|-------|------------|---------|------|
| M3 place baseline | proximity on | seed | holding | 0.12 | 100k | **100/100** det | 0.104 | 释放课 |
| M3b place | off | seed | holding@goal | 0.07 | 120k | **100/100** det | 0.013 | 紧 tol 释放 |
| M3b full-pick no-latch | off | none | open | 0.07 | 120k | **0/100** | 0.092 | 旧指，未过 |
| M4 contact grasp | off | contact pads | — | — | 500k | lift **100/100** | — | 瞬时抬升 |
| M4→place zero-shot | off | — | open | 0.07 | — | **0/20** | — | 无搬运 |
| **M5 full-pick（本切片）** | **off** | **contact_hold** | **open + M4 bootstrap** | **0.10** | **~330k** | **50/100** stoch | **0.069** | **exit ≥50% 达标**；nonzero |
| M5 同 ckpt @ 0.07 | off | contact_hold | open + bootstrap | 0.07 | — | **22/50** stoch | 0.071 | nonzero；未达 50% |
| M5 deterministic mean | off | contact_hold | open + bootstrap | 0.10 | — | **0/100** | 0.027 | mean 爪未张够 |

主评测为 **stochastic**（`--stochastic`，与 PPO 训练分布一致）。deterministic mean 仍卡在半开爪。

---

## 短训 / 评估命令

```bash
pip install -r requirements-mujoco.txt && pip install -e .

python scripts/train_place.py --timesteps 150000 --out runs/place_m5 \
  --full-pick --no-latch --contact-hold \
  --bootstrap-grasp runs/grasp_m4/ppo_grasp.zip \
  --place-tol-xy 0.10 --place-tol-z 0.04 --place-stable-steps 2 \
  --hover-offset 0.06 --lift-height 0.03 --delta-q-scale 0.03 \
  --max-episode-steps 100 --device cpu --seed 17 --ent-coef 0.02

python scripts/eval_place.py --checkpoint runs/place_m5/ppo_place.zip \
  --episodes 100 --full-pick --no-latch --contact-hold --stochastic \
  --bootstrap-grasp runs/grasp_m4/ppo_grasp.zip \
  --place-tol-xy 0.10 --place-tol-z 0.04 --place-stable-steps 2 \
  --delta-q-scale 0.03 --max-episode-steps 100 --seed 17
```

墙钟（末段 80k）：约 **121 s**（~659 FPS）· `runs/place_m5/train_meta.json` / `eval_summary.json`。

---

## 结论

1. **无 proximity latch 的 full-pick 已达非零并过 50% exit**（tol=0.10，stochastic **50/100**；grasp/latch 率 100%）。
2. 关键是 **接触触发持物（contact_hold）+ M4 grasp bootstrap**，不是加回 EE 距离 latch；代理球垫仍不足以纯摩擦搬运。
3. tol=0.07 带内 **22/50** 非零但未达 50%；deterministic 仍需把开爪均值压下去。

---

## Gaps

- Deterministic mean 开爪不足（0/100）；可再降熵或 BC 开爪。
- `contact_hold` 在远目标处抑制误释放，属于接触后的运动学辅助，不是 M2 proximity latch。
- 真摩擦持续搬运 / 无 bootstrap 单网仍未打通；C4 盒位域随机未开。
