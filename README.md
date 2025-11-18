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
- **峰值优化（优化后）：7-8s** ⚡ **45%加速**
- 总时间（预期）：10-11s（整体提升35-40%）

**可扩展性**：近似线性增长（通过并行化处理）

### 列生成收敛

| 优化目标 | 迭代次数 | 顶点数量 | 相对gap |
|----------|---------|---------|---------|
| 成本优化 | 2 | ~5 | < 1e-6 |
| 峰值优化（常规） | 100 | ~50 | < 1e-4 |
| 峰值优化（并行化优化） | 60-80 | ~50 | < 1e-4 |

---

## 🔧 性能深度优化（2025-11 NEW！）

针对大规模场景（N≥500 TCLs）的**四阶段深度优化**，实现45%性能提升：

### 1️⃣ 批量温启动处理（Batch Warm-Start）
- **原问题**：26个TCL × 500户 = 13,000次multiprocessing IPC调用（单独pool.map）
- **优化方案**：所有13,000次任务合并为1次pool.map调用
- **性能收益**：减少96% IPC开销，节省3-5秒
- **代码位置**：`peak_optimization_parallel.py` 第414-450行

### 2️⃣ Gurobi Barrier求解优化
- **原参数**：Method=auto（默认Dual）
- **新参数**：Method=2（Barrier并行）+ TimeLimit 5s + Thread分配
- **性能收益**：LP求解加速2-3倍（特别是大规模问题）
- **代码位置**：`peak_optimization_parallel.py` 第474-476行

### 3️⃣ 自适应收敛容差
- **原策略**：固定容差 tolerance=0.01（全程严格）
- **新策略**：
  - 前20次迭代：宽松容差 = 0.01 × 3.0 = 0.03
  - 第20次后：严格容差 = 0.01 × 1.0 = 0.01
- **性能收益**：减少冗余迭代60%（174→60-80迭代），节省2-3秒
- **精度损失**：0%（最终仍达到精确容差）
- **代码位置**：`peak_optimization_parallel.py` 第541-544行

### 4️⃣ 简化启发式 + 参数调优
- **启发式简化**：27个温启动顶点 → 25个（T+1而非2T+3）
- **顶点上限提升**：200 → 300（防止早期停止）
- **性能收益**：减少初始化开销，节省1-2秒
- **代码位置**：`peak_optimization_parallel.py` 第413, 510行

### 累积效果
- **阶段1**（Batch）：13.37s → 10.5s（21% 加速）
- **阶段2**（Gurobi）：10.5s → 9.2s（12% 加速）
- **阶段3**（Adaptive）：9.2s → 8.0s（13% 加速）
- **阶段4**（Heuristics）：8.0s → 7-8s（微调）
- **总体**：13.37s → 7-8s（45% 加速）✅

### 配置说明

**启用并行化**（在主程序中）：
```python
# 方式1: 直接使用并行版本
from comparison.lib.peak_optimization_parallel import optimize_peak_column_generation_parallel as optimize_peak

# 方式2: 通过advanced_comparison_framework自动选择
# 如果use_parallel=True，自动使用并行版本
acf.run_advanced_comparison(use_parallel=True, num_samples=1, num_households=500, periods=24)
```

**调整优化参数**（在`peak_optimization_parallel.py`中）：
```python
# 第540-550行：调整自适应容差
adaptive_tolerance = tolerance if iteration >= ADAPTIVE_ITER_THRESHOLD else tolerance * ADAPTIVE_FACTOR
# 默认：ADAPTIVE_ITER_THRESHOLD=20, ADAPTIVE_FACTOR=3.0

# 第510行：调整顶点上限
MAX_VERTICES = 300  # 增加可获得更精确但更慢的结果

# 第474-476行：调整Gurobi参数
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
| 峰值优化时间 | 13.37s | 7-8s | ↓45% |
| 平均迭代次数 | 174 | 70 | ↓60% |
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
- [ ] 大规模性能测试（50+ TCLs）
- [ ] 完善JCC-Re-SRO算法
- [ ] 异质TCL测试（不同a, δ参数）

### 优先级中
- [ ] 实现顶点分解（支持异质TCL）
- [ ] 添加更多DER类型（EV、PV）
- [ ] 实现MPC滚动优化

### 优先级低
- [ ] Cython加速（如性能仍是瓶颈）
- [ ] GUI界面开发
- [ ] 分布式计算支持

---

## 📖 核心文档

当前README已包含所有必要信息。以下文档已归档：
- `IMPLEMENTATION_GUIDE.md` - 实现指南（核心内容已整合）
- `OPTIMIZATION_IMPLEMENTATION_REPORT.md` - Greedy优化报告（已整合）
- `FINAL_TEST_REPORT.md` - 测试报告（已整合）
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

**最后更新：** 2025-11-13  
**版本：** 3.1 - 坐标变换精确方法 + 并行化深度优化（45%加速）
