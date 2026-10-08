# 当量子漫步遇见傅里叶：我用 500 行 Python 造了一个"动量平面引擎"

> 副标题：从一个物理幻想到 GitHub 开源项目的完整心路——周期性相干注入、离散时间量子漫步、以及它们在动量空间中绽放的衍射之美。

## 一、一切始于一个画面

闭上眼睛，想象这样一个场景：

在一片二维的空间晶格上，你每隔固定的距离放下一个"波包"——就像在水面上规律地投入石子。所有波包都朝着同一个方向传播，它们之间保持着完美的相位相干。然后，你让这些波包在晶格上做"量子跳跃"——每一步，它们都同时向四个方向探索，又在下一刻重新干涉。

最后，你做一次傅里叶变换。

奇迹发生了：原本在位置空间中散乱分布的波包，在动量空间中汇聚成一组锐利的、如同晶体衍射般的峰。这些峰的位置、数量、相对强度，精确地编码了你最初的注入几何和量子漫步的动力学参数。

这就是 **MomentumPlane**——一个我从物理畅想开始，一步步用代码实现的开源仿真引擎。

GitHub: https://github.com/eninem123/MomentumPlane

## 二、为什么这件事值得做？

说实话，这个项目的起点不是一个工程需求，而是一种**物理直觉的冲动**。

在量子光学中，周期性结构（比如光子晶体、衍射光栅）会在动量空间产生离散的模式。在量子计算中，离散时间量子漫步（DTQW）是量子算法的基础原型。这两件事之间有一个深刻的联系：**如果你在位置空间中周期性地注入相干态，然后让它做量子漫步，那么动量空间中的演化就是一部"干涉电影"。**

但现有的工具要么太学术（QuTiP 适合开放量子系统，但做 2D 晶格漫步不够直观），要么太工程（各种 FFT 库能做变换，但没有物理语义）。我想要的是一个**能让物理直觉直接变成视觉画面**的工具——拖一个滑块，就能看到动量平面的峰在分裂、汇聚、消散。

所以我自己写了一个。

## 三、架构：四个模块对应四步物理

整个项目的核心代码不到 500 行，但每一行都对应着清晰的物理语义。我把它拆成了四个模块：

### 3.1 Injector —— 周期性相干注入

```python
from momentum_plane import Injector

inj = Injector(grid_size=128, stride=16, wavevector=(0.8, 0.0), sigma=3.0)
psi0 = inj.inject()  # (128, 128) complex field
```

物理上，这是在二维晶格上放置一组高斯波包，每个波包带有相同的平面波相位 `exp(i(k·r))`。关键参数：

- **stride（注入间隔）**：决定了动量空间中峰的间距——间隔越大，峰越密（这是衍射光栅的基本性质）
- **wavevector（波矢）**：决定了整个动量平面的平移方向
- **sigma（波包宽度）**：决定了每个动量峰的宽度——位置空间越局域，动量空间越弥散（不确定性原理）

### 3.2 LatticeHop —— 离散时间量子漫步

这是整个项目物理上最硬核的部分。一个 2D DTQW 的每一步由两个操作组成：

**硬币算符（Coin）**：作用在内部自由度（4 个方向：上/右/下/左）上，是一个 4×4 的幺正矩阵。我实现了两种：

- **Hadamard 硬币**：平衡型，每个方向等概率叠加，产生对称的漫步
- **Grover 硬币**：扩散型，产生更聚焦的漫步模式

```python
# Hadamard coin: H2 ⊗ H2
H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
coin = np.kron(H2, H2)  # 4x4 unitary
```

**条件位移（Shift）**：根据硬币态的方向，将振幅移动到相邻格点。这就是"量子跳跃"——粒子不是经典地选择一个方向，而是同时向所有方向探索。

```python
# One DTQW step
psi = np.einsum("cd,xyd->xyc", coin, psi)  # coin
for d, (dx, dy) in enumerate(directions):
    psi[:, :, d] = np.roll(np.roll(psi[:, :, d], dx, axis=0), dy, axis=1)
```

整个演化是严格幺正的——总概率守恒。我写了单元测试来验证这一点。

### 3.3 FieldPlane —— 2D FFT 动量平面合成

```python
from momentum_plane import FieldPlane

field = FieldPlane(grid_size=128)
momentum = field.transform(position_field)  # 2D FFT + fftshift
intensity = field.intensity(position_field)  # |FFT|², normalised
```

这一步是整个项目的"魔法时刻"。2D FFT 将位置空间的波函数变换到动量空间。由于注入是周期性的，动量空间中出现的是一组离散的峰——就像 X 射线打在晶体上产生的劳厄斑。

我还实现了峰值检测（局部最大值 + 非极大值抑制），可以自动追踪演化过程中峰的数量和强度变化。

### 3.4 Visualizer —— 让物理看得见

```python
from momentum_plane import Visualizer

viz = Visualizer()
viz.plot_final_state(result)  # position + momentum side by side
viz.animate_evolution(result, output="evolution.gif")  # animated GIF
```

对数热力图、相位图、动画 GIF——这些不是装饰，而是**物理理解的工具**。当你看到动量平面的峰在 40 步内从模糊变得锐利，你就理解了什么叫"相位重对齐"。

## 四、从 Demo 到研究工具：JAX 后端

最初的 NumPy 实现已经能跑出漂亮的结果，但我很快意识到一个问题：**如果我想做参数扫描（比如扫描 100 个不同的波矢，看动量平面如何变化），NumPy 太慢了。**

于是我加了一个 **JAX 后端**。这不是简单的"把 numpy 换成 jax.numpy"——而是利用了 JAX 的三个核心能力：

### 4.1 JIT 编译

```python
@jax.jit
def step(psi, coin):
    psi = jnp.einsum("cd,xyd->xyc", coin, psi)
    # ... conditional shifts with jnp.roll
    return psi
```

第一次调用时编译，后续调用以原生速度运行。在 CPU 上，对于大网格（256×256 以上）和多步演化，编译后的循环比 NumPy 快 2-5 倍。

### 4.2 lax.scan 编译循环

普通的 Python for-loop 在 JAX 中会被展开（unroll），步数多了会导致编译时间爆炸。正确的做法是用 `jax.lax.scan`：

```python
def evolve(psi, n_steps):
    def body(carry, _):
        return step(carry, coin), None
    psi_final, _ = jax.lax.scan(body, psi, None, length=n_steps)
    return psi_final
```

这把整个演化循环编译成一个 XLA 计算图——在 GPU 上，这就是一个 kernel，没有 Python 开销。

### 4.3 vmap 批量参数扫描

这是我最喜欢的功能。想同时模拟 100 个不同的波矢？用 `vmap`：

```python
from momentum_plane.lattice_jax import batch_evolve

params = [{"wavevector": (kx, ky), "stride": 8} for kx, ky in wavevectors]
intensities = batch_evolve(params, grid_size=64, n_steps=30)
# intensities.shape == (100, 64, 64) — all simulated in parallel
```

在 GPU 上，这 100 个模拟是真正并行的。这对于做相图（phase diagram）研究来说是革命性的。

### 4.4 自动微分

JAX 的 `grad` 可以对物理参数求梯度。比如，你想知道"波矢 kx 变化多少会导致动量峰移动 1 个像素"——直接求导就行：

```python
def peak_position(kx):
    cfg = SimulationConfig(wavevector=(kx, 0.0), ...)
    result = MomentumPlanePipeline(cfg).run()
    return result["peak_positions"][0][0]  # x-coordinate of first peak

d_peak_d_kx = jax.grad(peak_position)(0.5)
```

这把仿真引擎从"可视化工具"升级成了"可微物理模拟器"——可以用来做参数优化、逆问题求解。

**关键设计决策**：JAX 后端和 NumPy 后端产生**完全相同的数值结果**（误差 < 1e-10）。我写了 12 个测试来验证这一点，包括 Hadamard/Grover 硬币、手性偏置、全流水线对比。用户可以放心切换后端。

## 五、工程规范：不只是"能跑"

一个开源项目要让人愿意用、愿意贡献，工程规范是底线。我做了这些：

### 5.1 36 个单元测试

覆盖幺正性验证、Parseval 能量守恒、概率守恒、形状契约、可复现性、NumPy/JAX 后端一致性。`pytest tests/ -v` 全绿。

### 5.2 GitHub Actions CI

每次 push 自动运行：
- **ruff** 代码风格检查
- **pytest** 在 Python 3.10/3.11/3.12 三个版本上跑测试
- **pytest-cov** 覆盖率报告，上传 Codecov

### 5.3 自动素材生成

这是一个有意思的工程 trick。GitHub 的 API 不支持直接上传二进制文件（PNG/GIF），但 README 又需要展示动图。我的解决方案是：写一个 GitHub Actions workflow，每次 push 时自动安装依赖 → 跑仿真 → 生成 PNG/GIF → 压缩 → commit 回仓库。这样 README 里的动图永远是最新代码生成的。

### 5.4 双语 README

英文为主（GitHub 默认），中文为 `README.zh-CN.md`，右上角互相跳转。物理引言的中英文版本都保留了原有的诗意。

## 六、Streamlit 交互面板

代码能跑是一回事，**让人愿意玩**是另一回事。我做了一个 Streamlit 面板，12+ 个滑块实时调参：

```bash
streamlit run app.py
```

- 注入器：网格大小、注入间隔、波矢 (kx, ky)、波包宽度、相位抖动
- 量子跳跃：演化步数、硬币类型、硬币相位偏置、边界条件
- 显示：记录间隔、相位图开关

拖动滑块，点击 Run，实时看到二维动量平面的变化。这是整个项目"可玩性"的核心——你不需要懂量子力学，也能通过调参感受到物理之美。

## 七、性能基准：诚实的数据

我写了一个基准脚本 `examples/benchmark.py`，对比 NumPy 和 JAX 在不同网格大小下的演化时间。

**诚实的结论**：在 CPU 上，对于小网格（< 128×128）和少步数（< 50），NumPy 实际上更快——因为 JAX 的 JIT 编译有固定开销，而 NumPy 的 C 级循环对小数组已经足够快。

JAX 的真正优势在三个场景：
1. **GPU**：安装 `jax[cuda]` 后，大网格上 10-100 倍加速
2. **批量参数扫描**：`vmap` 同时跑几百个配置
3. **自动微分**：对物理参数求梯度，做优化

这不是一个"JAX 永远更快"的营销话术——而是一个诚实的工程判断。

## 八、理论深度：为什么这不仅仅是"好看"

项目的 `docs/theory.md` 里有完整的数学推导，但核心思想可以用三句话概括：

1. **N 个相干注入波包的动量空间振幅**是单个波包傅里叶变换乘以一个"几何因子** `Σ exp(i k·r_n)`。当注入位置 `r_n` 是周期性的，这个几何因子就是一组 δ 函数——这就是动量峰的来源。

2. **DTQW 的色散关系** `ω(k)` 决定了每个动量分量的传播速度。不同动量分量以不同速度演化，导致位置空间的波包扩散，同时在动量空间中产生峰的展宽和分裂。

3. **相位抖动**是一个退相干参数。当注入波包之间的相位不再完美相干时，动量峰会变宽、变矮——这模拟了真实实验中的不完美性。

这些不是事后贴上去的"理论装饰"，而是代码实现的指导原则。每一个参数（stride、wavevector、sigma、coin_angle）都有明确的物理对应。

## 九、下一步：从这里走向哪里？

这个项目目前是 v0.1，但我对它的演进有清晰的路线图：

### 短期（已在做）
- ✅ JAX 后端（GPU + 自动微分 + vmap）
- ✅ CI/CD（ruff + pytest + coverage）
- ✅ Colab 一键运行
- 🔄 性能基准和优化

### 中期
- **Rust 核心**：用 Rust + PyO3 重写演化循环，目标 10-50 倍 CPU 加速。接口不变，用户无感升级。
- **3D 扩展**：从 2D 动量平面到 3D 动量体，用体渲染可视化
- **QuTiP 集成**：和主方程方法对比，研究开放量子系统中的动量平面演化

### 长期
- **可微物理模拟器**：基于 JAX 的自动微分，做逆问题——给定目标动量平面，反推注入参数
- **流式架构**：把量子漫步演化做成无锁计算管道，支持实时注入（这是最初畅想的"路线 B"）
- **论文**：把"周期性相干注入 + DTQW 在动量空间的衍射汇聚"这个现象整理成一篇短论文

## 十、写给想动手的你

如果你读到这里，可能会想："这个项目看起来很酷，但我能做什么？"

答案是：**任何你想做的。**

- 你是物理学生？用它来理解量子漫步和傅里叶光学，改参数看变化
- 你是工程师？用 JAX 后端做参数扫描，或者帮我写 Rust 核心
- 你是视觉设计师？帮我做更好的可视化——现在的热力图只是起点
- 你是研究者？用自动微分做逆问题，或者和你的实验数据对比

GitHub: https://github.com/eninem123/MomentumPlane

给个 Star，提个 Issue，或者只是跑一下看看——都是对这个项目的支持。

---

*"干涉，不过是秩序在傅里叶空间中与自己相遇。"*

---

**关于作者**：科技公司架构师，AI Agent 量化系统背景，对物理直觉和工程实现的交叉点有执念。博客：https://www.aialter.site
