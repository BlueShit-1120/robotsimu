# M9 — Isaac Lab Reach 短训（本机）

> 日期：2026-11-02 · 分支 `develop` · 前置 M7 env + M8 RL extras

## 命令

```bash
conda activate env_isaacsim
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
cd ~/IsaacLab
./isaaclab.sh -p /path/to/robotsimu/scripts/train_isaac_reach.py \
  --headless --timesteps 8192 --num-envs 4 --out runs/isaac_reach_m9 --seed 0
```

## 本机结果（2026-11-02）

| 项 | 值 |
|----|-----|
| ok | true |
| timesteps | 8192 |
| num_envs | 4 |
| learn 墙钟 | ~28 s |
| FPS | ~293 |
| obs/act | 31 / 8 |
| GPU | RTX 4070 Ti SUPER |
| checkpoint | `runs/isaac_reach_m9/ppo_isaac_reach.zip`（`runs/` gitignore） |

短训仅为训练闭环冒烟；未做成功率评测。加长训与 `eval` 为下一刀。
