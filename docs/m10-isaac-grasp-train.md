# M10 — Isaac Lab Grasp 短训（本机）

> 日期：2026-11-02 · 分支 `develop` · 前置 M7 env + M8 RL extras + M9 Reach

## 命令

```bash
conda activate env_isaacsim
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm
cd ~/IsaacLab
./isaaclab.sh -p /path/to/robotsimu/scripts/train_isaac_grasp.py \
  --headless --timesteps 8192 --num-envs 4 --out runs/isaac_grasp_m10 --seed 0
```

## 本机结果（2026-11-02）

| 项 | 值 |
|----|-----|
| ok | true |
| phase | grasp（ContactSensor on） |
| timesteps | 8192 |
| num_envs | 4 |
| learn 墙钟 | ~31 s |
| FPS | ~262 |
| obs/act | 31 / 8 |
| ep_rew_mean | 2.5 → **4.08** |
| explained_variance | −1.45 → ~0.5 |
| GPU | RTX 4070 Ti SUPER |
| checkpoint | `runs/isaac_grasp_m10/ppo_isaac_grasp.zip`（`runs/` gitignore） |

短训仅为训练闭环冒烟；无 latch / contact_hold；未做抬升成功率评测。加长训与 `eval` 为下一刀。
