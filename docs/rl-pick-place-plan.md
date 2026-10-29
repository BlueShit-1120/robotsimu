# 轮式人形机器人 RL 抓放方案与实施计划

> 项目：robot-simulation  
> 目标：基于 `robotsimu` **整机 URDF**，在 **MuJoCo** 与 **Isaac Lab** 双平台上完成工件抓放强化学习。  
> 状态：方案文档（仅计划，不实施代码）；**Phase 0 前置决策已锁定**  
> 资产盘点日期：2026-10-29（公开克隆 https://github.com/BlueShit-1120/robotsimu @ `6342d4c`）  
> SoT：[`robotsimu-inventory.md`](./robotsimu-inventory.md) · 增量：[`urdf-reanalysis-delta.md`](./urdf-reanalysis-delta.md) · 决策：[`decisions.md`](./decisions.md)

---

## 1. 当前资产盘点与就绪缺口

### 1.1 盘点结果（URDF SoT，已落地）

| 检查项 | 结果 |
|--------|------|
| 上游仓库 | https://github.com/BlueShit-1120/robotsimu 可公开克隆；HEAD `6342d4c`（`revise the robot urdf file`） |
| 内容 | `assets/DVT1/{urdf,meshes,mjcf}` + `assets/scenes/`；**无** sim_bridge / scene_manager / tools / 夹爪网格 |
| 模型格式（SoT） | **`assets/DVT1/urdf/robot_assemble.urdf`**：整机双臂 + 躯干/头 + 四舵四驱 + 双 Robotiq 2F-85 + FT300 |
| 参考 MJCF | `assets/DVT1/mjcf/robot_assemble.xml`（双臂同模；夹爪网格路径同样外挂） |
| xacro | 入口存在，但缺 `DVT1.urdf` 与 `../tools/*` → **不能在本仓重生成** |
| USD | **无** |
| 机体网格 | visual/collision 各 29 STL，URDF 引用命中；夹爪/F-T **22 文件缺失** |
| 本 Project `/workspace` | 空仓；资产未 submodule（默认只文档化） |

**机器人：** `spr_robot`（DVT1）；左右臂各 7 轴；腰系双臂安装；底座 URDF 无 freejoint（MJCF 有 `floating_base`）。

### 1.2 真实目录布局

```text
robotsimu/
├── README.md
└── assets/
    ├── DVT1/
    │   ├── urdf/robot_assemble.urdf   # ★ SoT
    │   ├── urdf/spr_robot*.urdf.xacro # 入口（依赖缺失）
    │   ├── meshes/{visual,collision}/
    │   └── mjcf/robot_assemble.xml
    └── scenes/                        # cnc* / bin_* YAML；CNC STL 已删
```

### 1.3 DOF / 夹爪 / 场景事实表

| 子系统 | DOF（URDF） | 名称 / 说明 |
|--------|-------------|-------------|
| 底座 | 0 | 锁 `base_link`；勿再假设旧 `base_freejoint` 焊死语义为唯一描述 |
| 舵 + 轮 | 4 + 4 | 有关节；**effort=velocity=0** → Phase0 冻死 |
| 躯干 / 腰 | 1 + 1 | `torso_lift_joint` \[0,0.7\] m；`waist_pitch_joint` \[0,π/4\]；场景常 0.5 / ~0.3 |
| 头 | 2 | yaw/pitch；冻 |
| 左臂 / 右臂 | **各 7** | `left/right_arm_joint1…7`；effort ±108/108/66/66/18/18/18 |
| **夹爪** | **每侧 1 驱动 + 5 mimic** | Robotiq 2F-85：`*_robotiq_85_left_knuckle_joint` \[0,0.8\]；effort 50 |

| 场景族 | 状态 |
|--------|------|
| CNC YAML | 仍在；fixture/peg/workpiece **STL 已删** → 不能直接复现 |
| `bin_*` YAML | 程序化料箱定义仍可读；旧 motions/suites/导纳档 **已删** |

**Phase 0 动作维（已锁定 D4）：** 右臂 7 + 右爪 1 = **8**（底盘/对侧臂/头锁死）。

### 1.4 针对 MuJoCo / Isaac Lab 的就绪缺口清单

就绪分：MuJoCo **~5.5/10**；Isaac Lab **~4.5/10**（相对旧 inventory：MuJoCo 因缺网格/场景下降；Isaac 因整机 URDF+夹爪上升）。细则见 inventory / delta。

#### 通用（两平台）

- [x] 整机双臂 URDF 同模；关节名稳定（`left/right_arm_joint*`）
- [x] 臂限位 + effort；夹爪 mimic 拓扑
- [x] 机体 mass/inertia 齐全；单位 m/kg/rad
- [x] **夹爪网格策略已定：代理几何（盒/柱）即可**（STL 非强制；见 [`decisions.md`](./decisions.md) D1）
- [ ] 实现代理指几何并修 mesh 相对路径（F-T 可视可省略或同代理）
- [ ] 碰撞简化（满 STL；舵无 collision）
- [x] **场景策略已定：自建桌面 + 长方体工件**（不依赖已删 CNC STL）
- [ ] 轮驱动参数（Phase 0+ 不需要；底盘冻结）

#### MuJoCo 特有

- [x] 已有整机 `robot_assemble.xml`（含夹爪 equality、noslip 经验注释）
- [ ] 用代理几何替换缺失夹爪 mesh 后验证 `MjModel` 可加载
- [ ] MJCF 夹爪关节名与 URDF `robotiq_85_*` 不一致 → 需映射层
- [ ] 场景注入器（旧 scene_manager / fixtures 不可用；自建桌面）
- [ ] XML 注释非法字符可能导致严格解析器失败

#### Isaac Lab / Isaac Sim 特有

- [ ] 本机安装 Isaac Sim（版本 **TBD by install**，见 D3）
- [ ] 无 USD；URDF→USD（代理几何可先导入）
- [ ] mimic 关节在 PhysX/Lab 的处理方式需钉版本验证
- [ ] 地面/灯光/桌面场景需自建
- [ ] CNC 毫米特征若恢复，GPU 接触仍高风险

### 1.5 最短复核命令

```bash
git clone https://github.com/BlueShit-1120/robotsimu.git && cd robotsimu
git rev-parse --short HEAD
python -c "import xml.etree.ElementTree as ET; r=ET.parse('assets/DVT1/urdf/robot_assemble.urdf').getroot();
print(r.get('name'), 'links', len(r.findall('link')), 'joints', len(r.findall('joint')))"
```

完整字段级结论：[`robotsimu-inventory.md`](./robotsimu-inventory.md)。

---

## 2. 推荐架构：共享任务 API + 平台适配器

原则：**任务语义一套，仿真后端两套**。先在 MuJoCo 上把「单臂 reach → grasp → place」跑通，再做 Isaac Lab 适配与 sim-to-sim 对齐，避免双平台同时改奖励导致不可控。

### 2.1 分层

```text
┌─────────────────────────────────────────────┐
│  training/          PPO/SAC 入口、超参、日志   │
├─────────────────────────────────────────────┤
│  tasks/pick_place/  观测/动作约定、奖励、终止   │  ← 平台无关
│                     curriculum 阶段定义       │
├─────────────────────────────────────────────┤
│  common/            关节名映射、坐标系、工件定义│
│                     成功判据、归一化、seed     │
├─────────────────────┬───────────────────────┤
│  backends/mujoco/   │  backends/isaac_lab/  │  ← 适配器
│  env, reset, step   │  env, reset, step     │
│  mjcf assets        │  usd / lab task cfg   │
└─────────────────────┴───────────────────────┘
│  assets/robotsimu/  符号链接或拷贝自上游资产   │
└─────────────────────────────────────────────┘
```

### 2.2 公共 vs 平台

| 归属 | 内容 |
|------|------|
| **common / tasks** | Obs/Act 规格；奖励；成功判据；curriculum；工件采样；**URDF 名 → 逻辑名**（`arm_r_j1`…、`grip_r`） |
| **backends/mujoco** | MJCF（修补网格后）、接触参数、Gymnasium 封装 |
| **backends/isaac_lab** | Lab 任务配置、USD、PhysX、向量化 env |
| **training** | 算法与超参；checkpoint |

### 2.3 接口约定（第一版建议）

- `reset(seed) -> obs`
- `step(action) -> obs, reward, terminated, truncated, info`
- `info` 至少含：`grasp_success`, `place_success`, `phase`, `ee_pose`, `object_pose`
- 动作：**臂关节增量（7）+ 夹爪开合（1）**；底盘/对侧臂第一阶段不暴露

---

## 3. RL 问题设计

### 3.1 任务定义（第一切片）— **已锁定**

见 [`decisions.md`](./decisions.md) D2 / D4。资产对齐：URDF **已有** 双臂 2F-85 拓扑；网格未入库 → **用代理几何**（D1）。

- **已选 B′：真实开合 grasp-and-place** — 右臂 7 + 右爪 1；桌面自由长方体；夹爪开合抓取 → 搬运 → 放置松开。  
- **A / C：** 不作默认；腕持压放与盒夹变体搁置。

**Phase 0 控制范围：** 冻结 `base_link`（轮/舵不暴露）；躯干/腰取场景常用固定值；左臂收纳；头锁定。训练动作维 = **右臂 7 + 夹爪 1 = 8**。

**工件规格（计划默认，可微调）：**

| 项 | 值 |
|----|-----|
| 形状 | 长方体 |
| 尺寸 | **5 × 4 × 3 cm** |
| 质量 | **0.15 kg**（区间 0.1–0.2 kg） |
| 摩擦 | μₛ ≈ **0.6**，μₖ ≈ **0.5**（指面/桌面同量级起步；C4 微随机） |

**场景：** 自建桌面 + 随机初始物位与臂姿态（限位内）；放置目标为桌面矩形区或固定位姿。不依赖已删 CNC STL。

### 3.2 观测空间（建议 v0）

优先 **本体感觉 + 物体状态**（无相机）：

| 分量 | 说明 |
|------|------|
| 臂关节位置 / 速度 | 归一化到 URDF 限位 |
| 夹爪开度 | 驱动 knuckle 角或归一化开合 |
| 末端位姿 | 相对底座：位置(3) + 旋转（6D/quat）；建议用 `right_arm_ee_link` / TCP |
| 工件位姿 | 位置 + 朝向 |
| 目标放置位姿 | 固定或相对 |
| 上一动作 / 阶段 one-hot | 可选 |

### 3.3 动作空间（建议 v0）

- **连续：** \(\Delta q\) 右臂 7 + 夹爪开合 1  
- **底盘 / 躯干 / 左臂：** 不暴露  
- 每步增量 clip；力矩按 URDF effort 饱和；夹爪用位置伺服（对齐 MJCF `position` actuator 思路）

### 3.4 奖励草图

稀疏 + 塑形，两平台共用权重表：

1. **到达：** \(-\|p_{ee}-p_{obj}\|\)  
2. **朝向：** 夹爪接近轴对齐  
3. **抓取：** 闭合 + 接触 + 抬升高度一次性奖励  
4. **搬运：** \(-\|p_{obj}-p_{goal}\|\)  
5. **放置：** 进入目标区且稳定 + 松开大额奖励  
6. **正则：** 动作平滑、超限、thrashing、掉落惩罚  

成功判据写入 `info`，与奖励解耦。

### 3.5 Episode 终止

| 类型 | 条件 |
|------|------|
| **成功** | `place_success`（位姿误差 < ε，静止 N 步） |
| **失败** | 掉落、严重自碰、超时、关节异常 |
| **截断** | 最大步数 |

### 3.6 Curriculum：grasp → place

| 阶段 | 目标 | 简化 | 退出标准（建议） |
|------|------|------|------------------|
| C0 Reach | EE 靠近抓取预位姿 | 物体固定 | 平均距离 < 2–3 cm |
| C1 Grasp | 闭合抬升 | 放置奖励关 | 抬升成功率 ≥ 70% |
| C2 Carry | 持物到放置上方 | 目标区放大 | 持物到达率 ≥ 60% |
| C3 Place | 松开并稳定 | 目标区收紧 | 完整成功率 ≥ 50% |
| C4 Domain | 位姿/摩擦/质量微随机 | 双平台对齐 | 差距可控 |

---

## 4. 技术栈与 Conda 环境策略

用户本机已有 **miniconda**。云端与本机可共用同名环境约定。

### 4.1 推荐组合

| 角色 | 选择 | 说明 |
|------|------|------|
| 快速迭代 | **MuJoCo 3.x** + **Gymnasium** | 轻量调试接触与奖励 |
| RL（MuJoCo） | **Stable-Baselines3**（PPO）或 CleanRL | 先出结果 |
| 规模化 / GPU | **Isaac Lab** | 向量化；后续视觉 |
| 可选 | skrl / rsl_rl | Isaac 生态 |

**第一切片算法：** PPO；SAC 留作接触稀疏备选。

### 4.2 Isaac Lab / Sim 安装（D3）

- **决定：** 在用户本机安装 Isaac Sim + 匹配 Isaac Lab。  
- **版本：** **TBD by install**（由 Local Isaac install worker 选定；选定后回填本小节并写入 `docs/env-pins.md`）。  
- 候选区间仍参考官方现行 4.2+/4.5 线，但以 install worker 实测为准。  
- 无 GPU：**只跑 MuJoCo**。

### 4.3 Conda 环境命名

```text
conda create -n rs-mujoco python=3.11 -y
# pip: mujoco, gymnasium, stable-baselines3, numpy, scipy, imageio, tensorboard

# Isaac：按官方脚本，环境名建议 rs-isaac-lab；禁止与 rs-mujoco 混装
```

公共代码保持纯 PyTorch/NumPy，两边 `pip install -e .`。

---

## 5. 分阶段实施计划与里程碑

### M0 — 资产接入与代理几何（决策已清）

- **做：** submodule/拷贝 `robotsimu@6342d4c`；**夹爪用盒/柱代理几何**（STL 非强制，D1）；验证 URDF/MJCF 可加载；自建桌面 + **5×4×3 cm / 0.15 kg** 长方体工件；冻结底盘与对侧臂（D4）；关节名映射（URDF `robotiq_85_*` ↔ MJCF `*_gripper_*`）。  
- **退出：** MuJoCo 中右臂+右爪（代理指）可视化无爆炸；工件可被夹持接触；Isaac 本机安装进度见 D3（版本 TBD）。

### M1 — 单臂 Reach

- Gymnasium/Lab env；obs/act v0；仅距离奖励；固定物体。  
- **退出：** PPO 评估距离达标；可视化可复现。

### M2 — Grasp

- 夹爪动作 + 抬升判据；C0→C1。  
- **退出：** 抬升成功率 ≥ 70%（≥100 eval ep）。

### M3 — Place

- 完整奖励；C2→C3。  
- **退出：** 完整 pick-place ≥ 50%；无系统性 thrashing。

### M4 — 双平台对等

- Isaac 复刻同一 obs/act/reward；分别训练；记录物理参数表。  
- **退出：** 两边可学习；差距可解释。

### M5 — 训练与固化

- 长训、种子、checkpoint、复现 README。  
- **退出：** 一键评估曲线；版本钉死。

---

## 6. 风险与缓解

| 风险 | 表现 | 缓解 |
|------|------|------|
| **夹爪网格缺失** | 无法裸加载 / 无接触几何 | **已决策：** 代理几何（D1）；接受视觉保真度折损 |
| **代理指接触偏差** | 抓取边界与真指不一致 | 记录摩擦/外形假设；真机前再换高保真 mesh |
| **URDF↔MJCF 命名差** | 策略/映射错关节 | 单一映射表；测试 assert 名称集合 |
| **满 STL 碰撞** | 慢、抖动 | 简化凸包；先降低接触刚度课程 |
| **轮 effort=0** | 误开移动 DOF 无效 | **已决策：** Phase 0 冻死底盘（D4） |
| **Sim-to-sim** | 一边能学一边崩 | 共享任务定义；不硬迁权重 |
| **旧场景 STL 删除** | CNC/bin 复现失败 | **已决策：** B′ 自建桌面+长方体（D2） |
| **xacro 不可再生** | 改夹爪变体困难 | 以 `robot_assemble.urdf` 为钉扎；变体非 Phase 0 范围 |
| **Isaac 版本未钉** | Lab API 漂移 | **TBD by install**（D3）；钉死后写 `env-pins.md` |

---

## 7. 仓库内建议先建的最小目录（暂不写业务代码）

```text
.
├── README.md
├── pyproject.toml
├── environment-mujoco.yml
├── docs/                    # 可从 Agent Store 同步本计划
├── assets/robotsimu/        # 上游或 symlink（含补丁网格）
├── src/robot_simu_rl/
│   ├── common/
│   ├── tasks/pick_place/
│   ├── backends/{mujoco,isaac_lab}/
│   └── training/
├── scripts/{check_assets,eval_pick_place}.py
└── configs/{task_pick_place,ppo_mujoco}.yaml
```

**本交付物范围：** 方案 / 盘点 / 决策文档；**不**实现代码。Phase 0 前置决策已锁定 → 可启动 M0（代理几何 + 桌面工件）。

---

## 8. 开放问题

### 已锁定（2026-10-29 用户决策 + 资产再分析）

| # | 问题 | 结论 |
|---|------|------|
| — | SoT 格式 | **整机 URDF** `robot_assemble.urdf` |
| — | DOF / 命名 | 双臂各 7；轮舵 effort=0 |
| — | 夹爪拓扑 | Robotiq 2F-85 + mimic；网格未进仓 |
| **D1** | 夹爪网格 | **非强制**；缺失则用盒/柱代理；接受视觉保真度折损 |
| **D2** | 任务 / 工件 | **真开合抓放**；长方体 **5×4×3 cm / 0.15 kg / μ≈0.6·0.5**；自建桌面 |
| **D3** | Isaac Sim | **本机安装**；版本 **TBD by install** |
| **D4** | Phase 0 范围 | **冻底盘**；训 **右臂 7 + 爪 1** |

### 仍待答复（不阻塞 M0 启动）

1. **资产合入方式：** submodule / vendor 拷贝 / 仅本地路径？  
2. **成功指标：** 抓放公差（mm/deg）、是否允许滑动、力上限？  
3. **视觉：** 第一版是否坚持本体感觉 + 物体状态？（方案默认是）  
4. **xacro 源：** 是否补交 `DVT1.urdf` + `tools/`（仅影响变体，非 Phase 0）？  
5. **Isaac 精确版本：** 待 Local Isaac install worker 回填（D3）。

---

## 9. 建议的立即下一步

1. Local Isaac install worker：本机安装 Isaac Sim，回填版本至 D3 / `env-pins.md`。  
2. 决定 `robotsimu` 合入方式后启动 **M0**（代理指 + 桌面长方体 + 冻底盘）。  
3. MuJoCo 冒烟：代理几何后加载 `robot_assemble`；右臂+爪可视化与接触。  
4. 实现骨架 + **右臂 Reach→Grasp→Place**（8 维动作）。

---

*文档维护：2026-10-29 锁定 D1–D4（见 [`decisions.md`](./decisions.md)）；里程碑退出标准可按实测略调，但勿在双平台同时改奖励公式。*
