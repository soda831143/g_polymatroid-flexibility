"""
诊断坐标变换的正确性
"""
import numpy as np

# 测试参数
a = 0.8
delta = 1.0
x0 = 0.0
x_plus = 3.0  # 物理状态约束
T = 5

print("="*80)
print("坐标变换诊断")
print("="*80)
print(f"参数: a={a}, delta={delta}, x0={x0}, x_plus=±{x_plus}")
print(f"时间步数: T={T}")

# 物理动力学验证
print("\n1. 物理动力学验证")
print("-"*80)
print("物理动力学: x(k+1) = a·x(k) + delta·u(k)")
print("展开: x(k) = a^k·x0 + delta·Σ_{i=0}^{k-1} a^(k-1-i)·u(i)")

# 测试控制序列
u_phys = np.array([1.0, -0.5, 0.8, -0.3, 0.5])
x_phys = np.zeros(T+1)
x_phys[0] = x0

for k in range(T):
    x_phys[k+1] = a * x_phys[k] + delta * u_phys[k]
    print(f"  k={k}: u={u_phys[k]:6.2f} → x[{k+1}] = {x_phys[k+1]:6.3f}")

# 虚拟坐标变换
print("\n2. 虚拟坐标变换")
print("-"*80)
print("虚拟状态: x̃(k) = x(k) / a^k")
print("虚拟控制: ũ(k) = delta·u(k) / a^k")

# 方案A: k从0开始计数（Python约定）
print("\n方案A: k从0开始（Python索引）")
print("  ũ[k] = delta·u[k] / a^k, x̃[k] = x[k] / a^k")

u_virt_A = np.zeros(T)
x_virt_A = np.zeros(T+1)
x_virt_A[0] = x0  # x̃[0] = x0 / a^0 = x0

for k in range(T):
    u_virt_A[k] = delta * u_phys[k] / (a ** k)
    x_virt_A[k+1] = x_phys[k+1] / (a ** (k+1))
    # 验证虚拟动力学
    x_virt_from_dynamics = x_virt_A[k] + u_virt_A[k]
    print(f"  k={k}: ũ[{k}]={u_virt_A[k]:8.3f}, x̃[{k+1}]={x_virt_A[k+1]:8.3f}, x̃[{k}]+ũ[{k}]={x_virt_from_dynamics:8.3f}, 误差={abs(x_virt_A[k+1]-x_virt_from_dynamics):.6f}")

# 方案B: k从1开始计数（TEX约定）
print("\n方案B: k从1开始（TEX公式）")
print("  ũ(k) = delta·u(k) / a^k, 其中k=1,2,...,T")
print("  Python: ũ[t] = delta·u[t] / a^(t+1), 其中t=0,1,...,T-1")

u_virt_B = np.zeros(T)
x_virt_B = np.zeros(T+1)
x_virt_B[0] = x0

for t in range(T):
    k = t + 1  # TEX中的时间索引
    u_virt_B[t] = delta * u_phys[t] / (a ** k)
    x_virt_B[t+1] = x_phys[t+1] / (a ** k)
    # 验证虚拟动力学
    x_virt_from_dynamics = x_virt_B[t] + u_virt_B[t]
    print(f"  t={t}(k={k}): ũ[{t}]={u_virt_B[t]:8.3f}, x̃[{t+1}]={x_virt_B[t+1]:8.3f}, x̃[{t}]+ũ[{t}]={x_virt_from_dynamics:8.3f}, 误差={abs(x_virt_B[t+1]-x_virt_from_dynamics):.6f}")

# 虚拟约束边界
print("\n3. 虚拟状态约束边界")
print("-"*80)
print("物理约束: -x_plus <= x(k) <= x_plus")

print("\n方案A: x̃(k) = x(k) / a^k")
for k in range(T+1):
    x_lower_virt = -x_plus / (a ** k) if k > 0 else -x_plus
    x_upper_virt = x_plus / (a ** k) if k > 0 else x_plus
    print(f"  k={k}: x̃ ∈ [{x_lower_virt:8.3f}, {x_upper_virt:8.3f}]")

print("\n方案B: x̃(k) = x(k) / a^k, k从1开始")
print("  Python x̃[t] = x[t] / a^(t+1), 但x̃[0] = x[0] (未缩放)")
for t in range(T+1):
    if t == 0:
        x_lower_virt = -x_plus  # x̃[0] = x[0]
        x_upper_virt = x_plus
    else:
        k = t
        x_lower_virt = -x_plus / (a ** k)
        x_upper_virt = x_plus / (a ** k)
    print(f"  t={t}: x̃ ∈ [{x_lower_virt:8.3f}, {x_upper_virt:8.3f}]")

# 累积约束（GeneralDER格式）
print("\n4. 累积约束（GeneralDER格式）")
print("-"*80)
print("GeneralDER: x_min[t] <= Σ_{s=0}^{t} ũ[s] <= x_max[t]")
print("这对应虚拟状态: x̃[t+1] = x̃[0] + Σ_{s=0}^{t} ũ[s]")

print("\n当前代码（可能有误）:")
for t in range(T):
    # 当前代码使用 a^(t+1)
    denom = a ** (t + 1)
    y_lower = -x_plus / denom - x0
    y_upper = x_plus / denom - x0
    print(f"  x_min[{t}] = {y_lower:8.3f}, x_max[{t}] = {y_upper:8.3f}")

print("\n修正方案（使用a^(t+1)对应x̃[t+1]）:")
print("  这是正确的！因为x_min[t]对应Σ_{s=0}^{t} ũ[s] = x̃[t+1] - x̃[0]")
print("  而x̃[t+1] = x[t+1] / a^(t+1)，所以约束为: -x_plus/a^(t+1) - x0 <= x̃[t+1] - x0 <= x_plus/a^(t+1) - x0")

print("\n=" * 80)
print("结论:")
print("=" * 80)
print("如果方案A和B的虚拟动力学都满足 x̃[k+1] = x̃[k] + ũ[k],")
print("那么当前的虚拟边界计算应该是正确的。")
print("问题可能在于逆变换或其他地方。")
