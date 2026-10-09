<div align="right">

[English](README.md) | [**简体中文**](README.zh-CN.md)

</div>

<div align="center">

[![CI](https://github.com/eninem123/MomentumPlane/actions/workflows/ci.yml/badge.svg)](https://github.com/eninem123/MomentumPlane/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-accelerated-9cf.svg)](https://github.com/google/jax)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eninem123/MomentumPlane/blob/main/examples/momentum_plane_demo.ipynb)

</div>

# 🌊 MomentumPlane — 动量平面引擎

> *"在动量空间的晶格上，每一次相干注入都是一次低语，每一次离散跳跃都是一次回响。当无数低语在傅里叶的尽头相遇，它们汇聚成光。"*

一个受量子光学与离散时间量子漫步启发的、用于模拟**周期性相干注入与离散量子跳跃**的高性能计算与可视化仿真框架。灵感来源于量子光学、离散时间量子漫步，以及当"秩序"遇见"干涉"时所绽放的令人屏息的衍射图案。

---

## ✨ 视觉效果

**位置空间 → 动量平面** — 观察散射波包如何坍缩为锐利的衍射峰（对角波矢，6 个动量峰）：

![Evolution](assets/diagonal_evolution_compressed.gif)

**最终态 — 位置空间概率密度（左）vs 动量平面强度（右）：**

![Final State](assets/diagonal_final.png)

**演化过程中的峰值收敛曲线：**

![Convergence](assets/diagonal_convergence.png)

**四种不同物理机制对比 — 经典、对角、Grover、手性：**

| 经典 Hadamard | 对角波矢 | Grover 硬币 | 手性偏置 |
|---|---|---|---|
| ![classic](assets/classic_final.png) | ![diagonal](assets/diagonal_final.png) | ![grover](assets/grover_final.png) | ![chiral](assets/chiral_final.png) |

> **演示素材由 GitHub Actions 自动生成**，每次 push 后自动重新渲染。本地运行 `python examples/generate_demo_assets.py` 可生成高分辨率版本。

---

## 🧠 30 秒看懂物理

1. **注入** — 在二维晶格上按固定空间间隔（子晶格）放置相干高斯波包，所有波包共享同一动量方向。这就是你的"源"。

2. **跳跃** — 通过**离散时间量子漫步（DTQW）**演化波函数：每一步施加 Hadamard（或 Grover）硬币算符，随后执行条件位移。漫步器在晶格上扩散，携带着相位信息传播。

3. **变换** — 做一次 2D FFT。规则排列的注入点相当于一个衍射光栅：在动量空间中，你会得到一组**锐利的峰梳**，其位置、宽度和相对强度编码了注入几何与量子漫步动力学的全部信息。

4. **汇聚** — 随着量子漫步的进行，相位重新对齐，动量峰变得更锐利、分裂、或形成焦散状结构。这就是"动量平面"——底层物理的视觉指纹。

---

## 🚀 快速开始

```bash
# 克隆
git clone https://github.com/eninem123/MomentumPlane.git
cd MomentumPlane

# 安装依赖
pip install -r requirements.txt

# 运行基础仿真（在 assets/ 目录生成 PNG + GIF）
python examples/basic_simulation.py

# 启动交互式面板
streamlit run app.py
```

### 最小示例（10 行代码）

```python
from momentum_plane import MomentumPlanePipeline, SimulationConfig

cfg = SimulationConfig(grid_size=64, stride=8, n_steps=30, seed=42)
result = MomentumPlanePipeline(cfg).run()

print(f"检测到的最终动量峰数: {result['n_peaks'][-1]}")
print(f"峰值强度: {result['peak_intensities'][-1]:.4f}")
```

---

## 🏗️ 架构设计

```
MomentumPlane/
├── momentum_plane/
│   ├── injector.py      # 周期性相干波包注入
│   ├── lattice.py       # DTQW（Hadamard/Grover 硬币 + 位移）— NumPy 后端
│   ├── lattice_jax.py   # DTQW — JAX 后端（jit, GPU, autodiff, vmap）
│   ├── boundary.py      # 几何边界掩码（圆环、多边形、条形、自定义）
│   ├── inverse_design.py # 梯度逆向设计（JAX 自动微分 + Adam 优化器）
│   ├── synthesizer.py   # 2D FFT 动量平面合成 + 峰值检测
│   ├── visualizer.py    # 热力图、相位图、动画 GIF 导出
│   └── pipeline.py      # 端到端编排（backend: numpy/jax）
├── examples/
│   ├── basic_simulation.py
│   ├── benchmark.py          # NumPy vs JAX 性能基准
│   ├── boundary_demo.py      # 拓扑边界约束演示
│   ├── inverse_design_demo.py # 逆向设计优化演示
│   ├── generate_demo_assets.py
│   ├── momentum_plane_demo.ipynb  # Google Colab 笔记本
│   └── wave_interference.py
├── tests/               # 76 个单元测试（幺正性、Parseval、NumPy/JAX 一致性、边界约束、逆向设计...）
├── docs/                # 理论推导文档
├── .github/workflows/   # GitHub Actions 自动生成演示素材
├── app.py               # Streamlit 交互式面板
└── requirements.txt
```

### 模块职责

| 模块 | 物理对应 | 代码实现 |
|--------|---------|------|
| **Injector** | 相干波包叠加 | 高斯包络 × 平面波相位，放置于子晶格 |
| **LatticeHop** | DTQW 幺正演化 | 4 方向硬币（C⁴）+ 条件位移，周期/反射边界 |
| **FieldPlane** | 动量空间衍射 | 2D FFT + fftshift，切趾窗，峰值检测 |
| **Visualizer** | 科学可视化 | 对数热力图，相位图，FuncAnimation GIF |

---

## 🎛️ 交互式面板

Streamlit 应用（`app.py`）允许你实时调节每一个物理参数：

- **注入器：** 网格大小、注入间隔、波矢 (kx, ky)、波包宽度、相位抖动
- **量子跳跃：** 演化步数、硬币类型（Hadamard/Grover）、硬币相位偏置、边界条件
- **显示：** 记录间隔、相位图开关

拖动滑块，点击 **Run**，实时观察动量平面的变化。

```bash
streamlit run app.py
```

---

## 🔷 拓扑边界约束

受**拓扑光子学**启发——波在晶格边界上单向无损传输，对缺陷免疫——MomentumPlane 支持几何边界掩码，将量子漫步束缚在特定形状内。

```python
from momentum_plane import MomentumPlanePipeline, SimulationConfig

# 圆环（环形波导）
cfg = SimulationConfig(
    grid_size=96, n_steps=35, seed=42,
    boundary_mask={"shape": "ring", "inner_radius": 15, "outer_radius": 35},
)
result = MomentumPlanePipeline(cfg).run()

# 六边形腔
cfg = SimulationConfig(
    boundary_mask={"shape": "polygon", "n_sides": 6, "radius": 30},
)

# 其他形状：圆形、三角形、正方形、条形（水平/垂直）、或自定义布尔掩码
```

| 自由传播 | 圆环约束 | 六边形腔 | 三角形腔 |
|---|---|---|---|
| ![free](assets/boundary_free.png) | ![ring](assets/boundary_ring.png) | ![hex](assets/boundary_hexagon.png) | ![tri](assets/boundary_triangle.png) |

> 左面板中的青色轮廓为边界掩码。运行 `python examples/boundary_demo.py` 可重新生成全部四组对比图。

---

## 🎯 逆向设计（梯度优化）

不再是从已知参数正向模拟，而是**指定目标动量平面，让优化器自动反解注入参数**。整条流水线——注入 → DTQW → FFT → 强度——通过 JAX 完全可微，梯度端到端流动。

```python
from momentum_plane.inverse_design import optimize, target_with_peaks, InverseDesignConfig

# 1. 定义目标（例如，在指定动量坐标处放两个峰）
target = target_with_peaks(64, [(20, 24), (44, 24)], peak_sigma=4.0)

# 2. 用 Adam 优化注入参数（kx, ky, sigma, phase_offset）
cfg = InverseDesignConfig(grid_size=64, stride=8, n_steps=20, n_iter=200, lr=0.02)
result = optimize(target, cfg)

# 3. 查看优化结果
print(result.params)       # {'kx': ..., 'ky': ..., 'sigma': ..., 'phase_offset': ...}
print(result.loss_history) # 收敛曲线
```

| 目标 | 初始（随机） | 优化后 | 损失曲线 |
|---|---|---|---|
| ![target](assets/inverse_design_peaks.png) | | | |

> 优化器自动发现能复现任意目标分布的波矢和波包宽度。支持 MSE 和相关系数两种损失函数。运行 `python examples/inverse_design_demo.py` 查看双峰和环形目标的完整演示。

---

## 📊 核心特性

- **物理严格** — 幺正演化验证、Parseval 能量守恒测试
- **双硬币算符** — Hadamard（平衡型）和 Grover（扩散型），支持手性相位偏置
- **灵活边界** — 周期性（环面）或反射性
- **拓扑约束** — 几何边界掩码（圆环、多边形、条形、自定义）束缚量子漫步，受拓扑光子学启发
- **逆向设计** — 全可微流水线（JAX 自动微分）+ Adam 优化器；指定目标动量，反解注入参数
- **动量峰检测** — 自动局部最大值检测 + 非极大值抑制
- **可复现** — 所有随机元素使用种子 RNG
- **76 个单元测试** — 覆盖幺正性、能量守恒、形状契约、NumPy/JAX 一致性、边界约束、梯度流动、优化收敛
- **交互式 Web UI** — Streamlit 面板，实时参数调节
- **动画 GIF 导出** — 完美适用于论文、演示和展示
- **CI/CD** — ruff 代码检查 + pytest-cov 覆盖率 + 自动素材生成

---

## 📐 理论文档

完整推导请参阅 [`docs/theory.md`](docs/theory.md)：

- N 个相干注入波包的动量空间振幅
- 为什么规则注入 → 衍射光栅 → 动量梳
- DTQW 色散关系及其对峰展宽的影响
- 相位抖动作为退相干参数

---

## 🛣️ 路线图

- [ ] **Rust 核心** — 用 Rust + PyO3 重写 `lattice.py` 演化循环（10-50 倍加速）
- [ ] **GPU 加速** — CuPy 后端，支持大网格（256×256 及以上）
- [ ] **3D 扩展** — 3D 晶格的动量体渲染
- [ ] **QuTiP 集成** — 将 DTQW 结果与主方程开放量子系统对比
- [ ] **在线演示** — 将 Streamlit 应用部署到 Community Cloud
- [ ] **C++ 参考实现** — 面向 HPC 集群

---

## 🤝 贡献

欢迎贡献！无论是新的硬币算符、更快的 FFT 后端、漂亮的可视化，还是理论修正——欢迎提 Issue 或 PR。

1. Fork 本仓库
2. 创建功能分支
3. 为新功能添加测试
4. 确保 `pytest tests/` 通过
5. 提交 PR

---

## 📄 许可证

[MIT](LICENSE) — 随便用，随便改，随便造。

---

## 🙏 致谢

灵感来源于量子光学之美、离散时间量子漫步之优雅，以及一个深刻的真理：**干涉，不过是秩序在傅里叶空间中与自己相遇。**

---

<div align="center">

**如果这个项目让你有所感触，给个 ⭐ 吧**

*用 NumPy、SciPy、Matplotlib 和无数个深夜的物理思考构建。*

</div>
