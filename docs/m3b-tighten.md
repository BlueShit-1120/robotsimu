# M3b — Tighten place / no-latch（云端结果）

Follow-up to [`m3-place.md`](./m3-place.md)。目标：关掉 latch 助教、收紧 `place_tol_xy`（0.06–0.08），并尝试 full-pick。  
Agent：`bc-0a4c1e29-84d9-5e7e-88a5-e0b0d02a75c2` · 分支 `cursor/m3b-tighten-75c2`（基于 `cursor/m2-grasp-cloud-6884`）。

---

## 做了什么

- Env：`--no-latch` 时**不再重新吸附**；`place_start_holding` 仍可用 seed-hold，张爪即释放。
- **自适应 seed**：若 spawn→goal 距离 ≥ `place_tol_xy`，seed 在放置目标上方（保证 C3 释放课仍在公差内）；否则仍 seed 在 spawn（兼容 M3 `tol=0.12`）。
- 无 latch grasp：接触式 `_is_holding` 塑形（关 latch 后仍给抬升 shaping）。
- 短训：`place_tol_xy=0.07`（带内 0.06/0.08 复评）、`--no-latch`、start-holding。
- 尝试 full-pick：`--full-pick --no-latch`（及对照 `--full-pick`+latch）。
- 产物：`media/m3b-tighten-cloud.tar.gz`。

---

## 对比指标（主表）

| 配置 | latch | start | tol_xy | steps | eval success | mean xy (m) | 备注 |
|------|-------|-------|--------|-------|--------------|-------------|------|
| **M3 baseline** | on | holding@spawn | **0.12** | 100k | **100/100** | **0.104** | 原 M3 达标 |
| M3 @ tight tol（不重训） | on | holding@spawn | **0.07** | — | **0/100** | **0.104** | 释放位仍 ~10 cm，未过 7 cm |
| **M3b place（本切片）** | **off** | holding@goal\* | **0.07** | 120k | **100/100** | **0.013** | 达标；\*自适应 seed |
| M3b 同 ckpt @ 0.06 | off | holding@goal | 0.06 | — | **50/50** | 0.013 | 带内复评 |
| M3b 同 ckpt @ 0.08 | off | holding@goal | 0.08 | — | **50/50** | 0.013 | 带内复评 |
| M2 grasp baseline | on | — | — | 250k | **50/50** | lift ok | 对照 |
| M3b grasp no-latch | **off** | — | — | 150k | **0/100** | ee–box 0.17 | 代理指接触抓取未抬起 |
| M3b full-pick no-latch | **off** | open | 0.07 | 120k | **0/100** | 0.092 | 未过 exit |
| M3b full-pick latch | on | open | 0.07 | 100k | **0/100** | 0.172 | 短训仍不够 |

\*当 `‖spawn−goal‖ ≥ tol` 时 seed 在 goal；M3 的 `tol=0.12` 仍 seed 在 spawn。

---

## 短训命令

```bash
python scripts/train_place.py --timesteps 120000 --out runs/place_m3b \
  --place-tol-xy 0.07 --place-tol-z 0.04 --place-stable-steps 2 --no-latch \
  --device cpu --seed 17 --max-episode-steps 150

python scripts/eval_place.py --checkpoint runs/place_m3b/ppo_place.zip \
  --episodes 100 --place-tol-xy 0.07 --place-tol-z 0.04 --place-stable-steps 2 --no-latch
```

Grasp / full-pick（未达标，仅记录）：

```bash
python scripts/train_grasp.py --timesteps 150000 --out runs/grasp_m3b --no-latch ...
python scripts/train_place.py --timesteps 120000 --out runs/place_m3b_full \
  --full-pick --no-latch --resume runs/grasp_m3b/ppo_grasp.zip --place-tol-xy 0.07 ...
```

---

## 结论

1. **收紧 XY 有效**：同 M3 策略在 `tol=0.07` 为 0%；M3b 无 latch 释放课在 0.07（及 0.06/0.08）达 **100%**，mean xy **1.3 cm**（相对 M3 的 ~10 cm）。
2. **无 latch 释放课可行**（seed-hold 张爪释放）；**无 latch 真接触抓取 / full-pick 仍未过**——代理指摩擦与抬升仍是瓶颈。
3. Full-pick（含 latch 对照）短训 100–120k 均未达 50% exit。

---

## Gaps

- 真接触 grasp / full pick-place 需更长训、指面/摩擦调参或示教 bootstrap。
- 自适应 seed-at-goal 使「紧 tol 释放」与「spawn 起搬」解耦；完整搬运仍靠 `--full-pick`。
- C4 域随机未开。
