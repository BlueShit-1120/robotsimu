# M8 — Isaac Lab RL extras 安装（本机）

> 日期：2026-11-02 · 分支 `develop`  
> 环境：conda `env_isaacsim` · Isaac Sim 4.5 · Lab v2.1.1（`~/IsaacLab`）

## 命令

```bash
conda activate env_isaacsim
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
cd ~/IsaacLab && ./isaaclab.sh --install all
```

## 结果

| 包 | 版本 |
|----|------|
| rsl-rl-lib | 2.3.3 |
| rl-games | 1.6.1 |
| skrl | 2.1.1 |
| stable-baselines3 | **2.7.1**（钉扎；2.9.x 要 torch≥2.8） |

`./isaaclab.sh --install all` 的 extras 阶段可能把 torch 升到 2.14.x。本机钉扎恢复：

```bash
pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install 'numpy<2' 'pillow==11.2.1' 'stable-baselines3>=2.6,<2.8'
```

终态：torch **2.7.0+cu128** · numpy **1.26.4** · pillow **11.2.1** · CUDA 可见 4070 Ti SUPER。
