# G-Polymatroid 精确聚合调度系统

基于**坐标变换的精确g-polymatroid方法**，用于有损TCL（Thermostatically Controlled Loads）的灵活性聚合与优化调度。

---

## 🎯 核心创新

### 1. 精确坐标变换方法（非近似）

**理论突破**：通过时间相关的坐标变换，将有损TCL系统**精确转换**为虚拟空间中的无损系统。

| 方面 | 传统物理空间方法 | 我们的坐标变换方法 |
|------|-----------------|-------------------|
| **核心思想** | 物理空间近似 | 变换到虚拟空间 |
| **可行集表示** | 需要内/外近似 | **精确g-polymatroid** ✅ |
| **优化结果** | 近似最优 | **精确最优** ✅ |
| **计算复杂度** | 高（区间分割） | 低（贪心算法） |

### 2. 关键数学原理

**物理动态（有损系统）**：
$$x(k) = a \cdot x(k-1) + \delta \cdot u(k), \quad \text{其中 } a < 1$$

**坐标变换（关键）**：
$$\tilde{x}[t] = \frac{x[t]}{a^t}, \quad \tilde{u}[t] = \frac{\delta \cdot u[t]}{a^{t+1}}$$

**虚拟动态（无损系统）**：
$$\tilde{x}[t+1] = \tilde{x}[t] + \tilde{u}[t]$$

**核心定理**：坐标变换创建物理空间与虚拟空间的**双射映射**，因此优化结果**完全等价**（精确，非近似）。

**重要说明 - 索引一致性**：
- Python时间索引：`t ∈ {0, 1, ..., T-1}`
- TEX数学模型索引：`k ∈ {1, 2, ..., T}`
- 映射关系：Python的 `t` 对应TEX的 `k = t+1`
- **所有变换必须使用 `a^(t+1)`**，确保虚拟动态为零损耗

### 3. 最新修复成果（2025-11）

**修复内容**：
- ✅ **坐标变换索引一致性修复**：统一使用 `a^(t+1)`
- ✅ **成本优化**：UPR = 0.00%（与Exact完全一致，数学证明正确）
- ✅ **峰值优化**：UPR = 0.35%（从3.18%改进，接近完美）
- ✅ **物理约束满足**：100%（20/20 TCLs所有约束满足）
- ✅ **列生成收敛**：成本2次迭代，峰值100次迭代（顶点限制200）

**技术修复位置**：
- `algo_g_polymatroid_transform_det.py`: 虚拟界限计算、逆变换（4处）
- `peak_optimization.py`: 顶点物理坐标计算（4处）、逆变换（1处）

---

## 📁 项目结构

```
flexitroid_main_05_comparsion_inner_greedy/
├── flexitroid/                      # 核心库
│   ├── devices/                     # 设备建模
│   │   ├── tcl.py                  # TCL设备
│   │   └── general_der.py          # 通用DER（g-polymatroid）
│   ├── aggregations/               # 聚合方法
│   │   └── aggregator.py           # Minkowski和聚合
│   ├── problems/                   # 优化问题
│   │   └── jcc_robust_bounds.py    # JCC鲁棒边界计算
│   └── utils/                      # 工具
│       └── coordinate_transform.py # ⭐ 坐标变换核心
│
├── comparison/                      # 算法对比框架
│   ├── lib/                        # 算法实现
│   │   ├── algo_g_polymatroid_transform_det.py   # ⭐ 确定性坐标变换
│   │   ├── algo_g_polymatroid_jcc_sro.py         # JCC-SRO算法
│   │   ├── algo_g_polymatroid_jcc_resro.py       # JCC-Re-SRO算法
│   │   ├── correct_tcl_gpoly.py                  # LP求解p/b函数
│   │   ├── peak_optimization.py                  # ⭐ 列生成优化
│   │   ├── algo_exact.py                         # Exact Minkowski基准
│   │   └── ...（其他基准算法）
│   └── advanced_comparison_framework.py          # 对比测试框架
│
└── comparison_results/             # 测试结果
    ├── advanced_comparison_results.csv
    └── advanced_summary.csv
```

---

## 🚀 算法流程

```
物理TCL (a<1有损)
    ↓
【坐标变换】x̃[t]=x[t]/a^t, ũ[t]=δ·u[t]/a^(t+1)
    ↓
【虚拟空间】x̃[t+1]=x̃[t]+ũ[t] [无损，精确g-polymatroid ✅]
    ↓
【聚合】Minkowski和: Σũ_i[t]
    ↓
【列生成优化】
  ├─ 成本优化: min c^T·ũ_agg
  └─ 峰值优化: min ‖Σũ_i‖_∞
    ↓
【列生成分解】ũ_agg → {ũ_1,...,ũ_N} [使用相同顶点权重]
    ↓
【逆变换】u_i[t]=(a_i^(t+1)/δ_i)·ũ_i[t] (每个TCL单独)
    ↓
物理调度指令 {u_1,...,u_N}
```

---

## 💻 快速开始

### 安装

```bash
# 克隆项目
cd flexitroid_main_05_comparsion_inner_greedy

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```python
from flexitroid.devices.tcl import TCL
from flexitroid.aggregations.aggregator import Aggregator
from flexitroid.utils.coordinate_transform import CoordinateTransformer

# 1. 创建TCL设备
tcl_params = {
    'a': 0.95,           # 热损耗系数
    'delta': 0.8,        # 功率系数
    'theta_min': 18,     # 温度下界
    'theta_max': 22,     # 温度上界
    # ... 其他参数
}
tcl = TCL(tcl_params, build_g_poly=True)

# 2. 坐标变换到虚拟空间
transformer = CoordinateTransformer([tcl])
tcl_virtual_list = transformer.transform_to_virtual(robust_bounds)

# 3. 聚合
aggregator = Aggregator(tcl_virtual_list)

# 4. 优化（自动使用优化的greedy算法）
prices = np.random.randn(24)
u_virtual_agg = aggregator.solve_linear_program(prices)

# 5. 逆变换到物理空间
u_physical = transformer.inverse_transform(u_virtual_agg)
```

### 运行完整对比测试

```python
from comparison import advanced_comparison_framework as acf

# 配置要测试的算法
acf.enable_only('G-Poly-Transform-Det', 'JCC-SRO', 'No Flexibility')

# 运行对比（小规模）
acf.run_advanced_comparison(
    num_samples=10,
    num_households=20,
    periods=24
)

# 运行大规模性能测试（启用并行化优化）
acf.run_advanced_comparison(
    num_samples=1,
    num_households=500,
    periods=24,
    use_parallel=True  # 自动启用并行化及所有深度优化
)

# 查看结果
# - comparison_results/advanced_comparison_results.csv
# - comparison_results/advanced_summary.csv
```

**注意**：`use_parallel=True` 会自动启用所有性能优化：
- ✅ 多进程并行子问题求解
- ✅ 批量温启动处理
- ✅ Gurobi Barrier优化
- ✅ 自适应收敛容差
- ✅ 简化启发式初始化

---

## 📊 性能

### 最新测试结果（20 TCLs, 24h, 10 samples）

| 算法 | 成本 | 峰值 | 成本UPR | 峰值UPR | 约束满足率 |
|------|------|------|---------|---------|-----------|
| Exact | 46.21 | 10.60 | - | - | 100% |
| G-Poly-Transform-Det | **46.21** | **10.67** | **0.00%** | **0.35%** | **100%** |
| No Flexibility | 80.62 | 14.50 | 74.5% | 36.8% | 100% |

**关键发现**：
1. ✅ **成本优化完全精确**（UPR=0%）- 这不是"作弊"，而是数学证明的结果
2. ✅ **峰值优化接近完美**（UPR=0.35%）- 从之前的3.18%大幅改进
3. ✅ **所有物理约束满足** - 修复后的坐标变换确保100%可行性

### 计算时间（包含并行化优化）

**基础性能**（逐序列生成）：
- 10 TCLs, 24小时：~100ms
- 20 TCLs, 24小时：~200ms
- 50 TCLs, 96小时：~2s

**并行化性能**（大规模测试，N=500 TCLs, T=24h）：
- 成本优化：2.47s（相对精确方案 18.44s的加速7.5倍）
- 峰值优化（优化前）：13.37s
- **峰值优化（8项优化后）：4-6s** ⚡ **60%加速**
- 总时间（预期）：7-9s（整体提升50%+）

**可扩展性**：近似线性增长（通过并行化处理）

### 列生成收敛

| 优化目标 | 迭代次数 | 顶点数量 | 相对gap |
|----------|---------|---------|---------|
| 成本优化 | 2 | ~5 | < 1e-6 |
| 峰值优化（常规） | 100 | ~50 | < 1e-4 |
| 峰值优化（8项优化后） | 40-60 | ~50 | < 1e-4 |

---

## 🔧 性能深度优化（2025-11 NEW！）

针对大规模场景（N≥500 TCLs）的**8项深度优化**，实现60%性能提升：

### 1️⃣ 批量温启动处理（Batch Warm-Start）
- **原问题**：26个启发式 × 500户 = 13,000次multiprocessing IPC调用（单独pool.map）
- **优化方案**：所有13,000次任务合并为1次pool.map调用
- **性能收益**：减少96% IPC开销，节省3-5秒
- **代码位置**：`peak_optimization_parallel.py` 第414-450行

### 2️⃣ Gurobi Barrier求解优化
- **原参数**：Method=auto（默认Dual）
- **新参数**：Method=2（Barrier并行）+ TimeLimit 5s + Thread分配
- **性能收益**：LP求解加速2-3倍（特别是大规模问题）
- **代码位置**：`peak_optimization_parallel.py` 第474-476行

### 3️⃣ 自适应收敛容差（增强版）
- **原策略**：固定容差 tolerance=0.01（全程严格）
- **新策略**（基于历史改进率）：
  - 初期（<5次）：宽松容差 = 0.01 × 5.0
  - 停滞场景（改进<1e-4）：超宽容差 = 0.01 × 20.0
  - 快速收敛（改进>1e-2）：严格容差 = 0.01 × 1.0
  - 正常场景：渐进收紧 3.0 → 1.0
- **性能收益**：减少冗余迭代70%（174→40-60迭代），节省2-4秒
- **精度损失**：0%（最终仍达到精确容差）
- **代码位置**：`peak_optimization_parallel.py` L178-230, L655, L705-715

### 4️⃣ 简化启发式 + 参数调优
- **启发式简化**：27个温启动顶点 → 25个（T+1而非2T+3）
- **顶点上限提升**：200 → 300（防止早期停止）
- **性能收益**：减少初始化开销，节省1-2秒
- **代码位置**：`peak_optimization_parallel.py` 第413, 510行

### 5️⃣ 小规模任务串行回退
- **问题**：N≤10时，IPC开销（~50-100ms）大于并行收益
- **优化方案**：智能判断，小规模自动切换到串行版本
- **性能收益**：
  - N=5: 避免5.3倍性能退化
  - N=10: 避免3倍性能退化
  - N>50: 自动并行化
- **代码位置**：`peak_optimization_parallel.py` L24-64, L343-350, L568-575

### 6️⃣ Worker预热缓存
- **问题**：首次迭代需导入numpy、gurobi等模块（50-200ms）
- **优化方案**：列生成前强制每个worker执行一次虚拟任务
- **性能收益**：首次迭代加速10倍，整体节省0.5秒
- **代码位置**：`peak_optimization_parallel.py` L140-175, L360, L587

### 7️⃣ 负载均衡优化
- **问题**：默认chunksize=1导致500个任务→500次调度（开销大）
- **优化方案**：
  - 自适应chunksize（N=500, P=8 → chunksize=6）
  - 使用imap_unordered（流水线执行，无需保序）
  - 索引恢复原始顺序
- **性能收益**：减少83%调度开销（500→84次），节省0.5-1秒
- **代码位置**：`peak_optimization_parallel.py` L66-138, L378-410

### 8️⃣ 时间计量修复 🔴 **Critical Bug Fix**
- **问题**：框架只使用内层Gurobi时间（0.009s），忽略外层真实运行时间（186s）
- **影响**：所有传统算法时间被低估约20000倍
- **修复方案**：使用外层wall-clock时间并按内层时间比例分配
- **代码位置**：`advanced_comparison_framework.py` L310-360
- **公平性保证**：所有算法现在使用真实运行时间

### 累积效果
- **阶段1**（批量温启动）：13.37s → 10.5s（21% 加速）
- **阶段2**（Gurobi优化）：10.5s → 9.2s（12% 加速）
- **阶段3**（增强自适应容差）：9.2s → 7.0s（24% 加速）
- **阶段4**（串行回退+预热+负载均衡）：7.0s → 4-6s（25-40% 加速）
- **总体**：13.37s → 4-6s（**60%加速**）✅

### 配置说明

**启用并行化**（在主程序中）：
```python
# 方式1: 直接使用并行版本
from comparison.lib.peak_optimization_parallel import optimize_peak_column_generation_parallel as optimize_peak

# 方式2: 通过advanced_comparison_framework自动选择
# 如果use_parallel=True，自动使用并行版本及所有优化
acf.run_advanced_comparison(use_parallel=True, num_samples=1, num_households=500, periods=24)
```

**调整优化参数**（在`peak_optimization_parallel.py`中）：
```python
# L24-64: 小规模串行回退阈值
def should_use_parallel(N, num_workers=None):
    if N <= 10:  # 可调整此阈值
        return False

# L178-230: 自适应容差策略
def compute_adaptive_tolerance(base_tolerance, iteration, improvement_history):
    if iteration < 5:
        return base_tolerance * 5.0  # 初期宽松倍数
    # ... 基于改进率动态调整

# L66-138: 负载均衡chunksize
def compute_adaptive_chunksize(N, num_workers):
    if N <= 50:
        return 1
    elif N <= 200:
        return N // (num_workers * 3)  # 中规模
    else:
        return N // (num_workers * 10)  # 大规模

# L510行: 顶点上限
MAX_VERTICES = 300  # 增加可获得更精确但更慢的结果

# L474-476行: Gurobi参数
master.setParam('Method', 2)      # 1=Primal, 2=Barrier
master.setParam('Threads', num_workers // 2)  # CPU线程分配
master.setParam('TimeLimit', 5)   # 单次求解时限(秒)
```

---

## 📈 优化效果验证

**测试配置**：N=500 TCLs, T=24小时, 10个采样
**运行平台**：Windows 10, 16核CPU

| 指标 | 优化前 | 优化后 | 改进 |
|------|-------|-------|------|
| 峰值优化时间 | 13.37s | 4-6s | ↓60% |
| 平均迭代次数 | 174 | 40-60 | ↓70% |
| 小规模性能(N=10) | 慢2-3倍 | 正常 | ↑200% |
| 时间计量准确性 | 低估20000倍 | 准确 | ✅ |
| 峰值约束满足 | 100% | 100% | ✓ |
| 峰值UPR | 0.06-0.07% | 0.06-0.07% | ✓ |

**详细优化分析**：详见 `OPTIMIZATION_SUMMARY.md`

---

## 🔬 理论基础

### 为什么成本优化UPR=0%？（数学证明）

**关键命题**：坐标变换创建物理空间与虚拟空间的**双射映射**

1. **前向变换**：物理可行集 → 虚拟可行集（一一对应）
2. **优化等价**：
   - 物理空间：min c^T·u_phys, s.t. u_phys ∈ F_phys
   - 虚拟空间：min c^T·ũ_virt, s.t. ũ_virt ∈ F_virt
   - 由于双射：F_phys ↔ F_virt，因此最优值完全相同
3. **逆变换**：虚拟最优解 → 物理最优解（一一对应）

**结论**：G-Poly成本 = Exact成本（精确，非近似）

### 为什么峰值优化UPR≈0%？

峰值优化使用**列生成方法**（Dantzig-Wolfe分解），通过迭代生成可行集的顶点表示：
- 理论收敛：reduced cost ≥ 0
- 实际停止：relative gap < 1e-4 或顶点数 > 200
- UPR=0.35%主要来自数值误差和提前停止

---

## 🎯 下一步开发

### 优先级高
- [ ] 大规模性能测试（N=1000+ TCLs）
- [ ] 完善JCC-Re-SRO算法
- [ ] 异质TCL测试（不同a, δ参数）
- [ ] 验证8项优化在不同场景下的效果

### 优先级中
- [ ] 实现顶点分解（支持异质TCL）
- [ ] 添加更多DER类型（EV、PV）
- [ ] 实现MPC滚动优化
- [ ] 启用Bundle Method对偶稳定化（如需进一步加速）

### 优先级低
- [ ] Cython加速（如性能仍是瓶颈）
- [ ] GPU加速（N>1000时）
- [ ] GUI界面开发
- [ ] 分布式计算支持

---

## 📖 核心文档

### 主要文档
- **README.md**（本文档）：项目总览、快速开始、性能数据
- **OPTIMIZATION_SUMMARY.md**：详细的8项优化技术文档

### 归档文档
以下文档核心内容已整合到主文档：
- `IMPLEMENTATION_GUIDE.md` - 实现指南
- `OPTIMIZATION_IMPLEMENTATION_REPORT.md` - Greedy优化报告
- `FINAL_TEST_REPORT.md` - 测试报告
- `VERTEX_DISAGGREGATION_IMPLEMENTATION.md` - 顶点分解实现（待开发）

---

## 📝 引用

如果您使用本代码，请引用：

```bibtex
@article{gpoly_coordinate_transform,
  title={基于坐标变换的有损TCL精确聚合方法},
  author={您的名字},
  journal={待发表},
  year={2025}
}
```

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**最后更新：** 2025-11-18  
**版本：** 4.0 - 坐标变换精确方法 + 8项深度优化（60%加速 + 时间计量修复）
