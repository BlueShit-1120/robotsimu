# M3 — Place（云端结果）

对齐 [`rl-pick-place-plan.md`](./rl-pick-place-plan.md) §5 M3 / curriculum **C2–C3**，决策 D1–D4。  
基于 M2 Grasp 树扩展 `phase="place"`。云端 agent：`bc-bcb9b44f-0ae6-592d-a2cc-823e4e396884` · 分支 `cursor/m2-grasp-cloud-6884`。

---

## 做了什么

- Env：`phase="place"` — 抓取/持物 → 搬运到放置目标 → 松开并稳定。
- **短训默认** `place_start_holding=True`：reset 时已 latch+微抬升（C2 入口），聚焦 C3 释放；`--full-pick` 可从开爪起步。
- 放置目标短训：`(0.70, -0.12)`（相对 spawn `(0.70,-0.20)` 约 8 cm）；`place_tol_xy=0.12`，`place_tol_z=0.04`，稳定 2 步。
- 继续使用 grasp **latch**（`--no-latch` 可关）。
- 脚本：`scripts/train_place.py`、`scripts/eval_place.py`；配置：`configs/ppo_place.yaml`。
- 产物：`media/m3-place-cloud.tar.gz`。

---

## 短训结果（已跑通）

| 项 | 值 |
|----|-----|
| 算法 | SB3 PPO · CPU |
| timesteps | **100_000**（release 课程） |
| 墙钟 | **~126 s**（~792 FPS） |
| start holding / latch | **on** |
| place_tol_xy / z | **0.12 m** / **0.04 m** |
| checkpoint | `runs/place_m3/ppo_place.zip` |
| eval | **100/100** place success，exit ≥50% **达标** |
| 摘要 | `runs/place_m3/eval_summary.json` |

```bash
python scripts/train_place.py --timesteps 100000 --out runs/place_m3 \
  --place-tol-xy 0.12 --place-tol-z 0.04 --place-stable-steps 2 --device cpu
python scripts/eval_place.py --checkpoint runs/place_m3/ppo_place.zip \
  --episodes 100 --place-tol-xy 0.12 --place-tol-z 0.04
```

---

## 成功判据

`info.place_success`：盒 XY 距目标 &lt; `place_tol_xy`，高度近桌面，**已释放**（非 latch），夹爪偏开，连续 `place_stable_steps` 步。

---

## 同步

与 M2 相同目标目录 `~/Desktop/spider-robot-sim/src/robotsimu`：

```bash
tar -xzf /path/to/m3-place-cloud.tar.gz --strip-components=1
# 或从分支 cursor/m2-grasp-cloud-6884 拉取后解压 media 包中的 runs/place_m3
python scripts/eval_place.py --checkpoint runs/place_m3/ppo_place.zip --episodes 100
```

协调器可从 **`/cursor/stores/user/docs/m3-place.md`** 与 **`/cursor/stores/user/media/m3-place-cloud.tar.gz`** 拾取。

---

## Gaps

- 短训放置目标较近、tol 偏宽；完整 25 cm 搬运需续训收紧目标。
- `--full-pick`（含 grasp）尚未在本切片达标。
- latch 仍为代理指助教。
