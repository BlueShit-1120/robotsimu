# M2 — Grasp（云端结果）

对齐 [`rl-pick-place-plan.md`](./rl-pick-place-plan.md) §5 M2 / curriculum **C1**，决策 D1–D4。  
从 M0（GitHub `BlueShit-1120/robotsimu`）重建 M1 Reach，再实现 Grasp。  
云端 agent：`bc-bcb9b44f-0ae6-592d-a2cc-823e4e396884` · 分支 `cursor/m2-grasp-cloud-6884`。

---

## 做了什么

- **Reach 重建（M1）**：`phase="reach"`，盒上 hover，夹爪 hold-open，右臂 MJCF **position** 伺服；脚本 `train_reach.py` / `eval_reach.py`。
- **Grasp（M2 / C1）**：`phase="grasp"`，夹爪可动；接近 → 闭合 → 抬升；放置奖励关闭。
- **Curriculum latch**（短训默认开）：EE–盒距离 &lt; `latch_dist`（0.12 m）且夹爪闭合时，对工件做运动学吸附，避免代理指接触过脆导致无法抬升。可用 `--no-latch` 关。
- 盒 spawn `(0.70, -0.20)`；grasp home 靠近工件；抬升阈值默认 **3 cm**。
- 脚本：`scripts/train_grasp.py`、`scripts/eval_grasp.py`；配置：`configs/ppo_grasp.yaml`。
- 产物包：`media/m2-grasp-cloud.tar.gz`。

---

## 短训结果（已跑通）

| 项 | 值 |
|----|-----|
| 算法 | SB3 PPO · CPU torch |
| timesteps | **250_000** |
| 墙钟 | **~349 s**（~716 FPS） |
| latch | **on**（默认） |
| hover / lift | **0.06 m** / **0.03 m** |
| checkpoint | `runs/grasp_m2/ppo_grasp.zip` |
| eval | **100/100** grasp success（抬升），exit ≥70% **达标** |
| 摘要 | `runs/grasp_m2/eval_summary.json` |

附带 Reach 冒烟短训（重建）：`runs/reach_m1/` · eval **10/10** · mean dist **0.048 m**（阈值 5 cm，hover 0.30）。

```bash
cd ~/Desktop/spider-robot-sim/src/robotsimu   # 同步后
# 或本云仓根目录
pip install -r requirements-mujoco.txt && pip install -e .

python scripts/train_grasp.py --timesteps 250000 --out runs/grasp_m2 \
  --hover-offset 0.06 --lift-height 0.03 --delta-q-scale 0.05 --device cpu \
  --max-episode-steps 120

python scripts/eval_grasp.py --checkpoint runs/grasp_m2/ppo_grasp.zip \
  --episodes 100 --hover-offset 0.06 --lift-height 0.03
```

---

## 成功判据

`info.grasp_success`（latch 模式）：已吸附 + 夹爪张开度 &gt; 0.25 + 盒高 &gt; spawn_z + `lift_height`。  
`info` 另含：`latched`、`reach_dist`、`reach_target`、`phase="grasp"`。

---

## 同步到本机 robotsimu

目标目录：`~/Desktop/spider-robot-sim/src/robotsimu`。

### A. 从 Git 分支（推荐，若已 Create repo / push 可读）

```bash
cd ~/Desktop/spider-robot-sim/src/robotsimu
git fetch <cloud-remote> cursor/m2-grasp-cloud-6884
git checkout -B m2-grasp cursor/m2-grasp-cloud-6884   # 或 merge/cherry-pick
# 拷贝 checkpoints（若未进 git）：
# 从 media 包解压 runs/
```

### B. 从 Project store 产物包

```bash
# 在 Cursor Project / Agent Store 下载：
#   docs/m2-grasp.md
#   media/m2-grasp-cloud.tar.gz

mkdir -p ~/Desktop/spider-robot-sim/src/robotsimu
cd ~/Desktop/spider-robot-sim/src/robotsimu
tar -xzf /path/to/m2-grasp-cloud.tar.gz
# 包内含 src/scripts/configs/assets 补丁 + runs/{grasp_m2,reach_m1} checkpoints
pip install -r requirements-mujoco.txt && pip install -e .
python scripts/eval_grasp.py --checkpoint runs/grasp_m2/ppo_grasp.zip --episodes 100
```

若本机已有 M1 Reach（`cursor/m1-reach-f7a0`），保留其 `runs/reach_m1`，用本包覆盖/合并 Grasp 相关文件即可。

---

## Gaps → Place（M3）

- latch 为短训助教；真接触抓取需调指面/摩擦或关 `--no-latch` 再训。
- 抬升阈值 3 cm（方案抬升成功率口径）；放置与搬运未开。
- 盒位固定；C4 域随机未做。
- Reach hover 短训仍 0.30 m；与 grasp 的 0.06 m 课程衔接可做两阶段 resume。
