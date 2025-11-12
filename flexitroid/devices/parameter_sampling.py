import numpy as np
'''
定义了几个参数采样函数，用于生成不同类型DER的参数。
这些函数使用随机数生成器来创建功率和能量约束，并返回这些约束的数组。
这些函数可以用于创建不同类型DER的实例，例如PV、V1G、V2G、E1S和E2S。
包含用于为各种DER设备（通用DER, PV, V1G, E1S, V2G, E2S）生成示例参数的函数。
这些函数通常使用随机采样（np.random.uniform 或 np.random.randint）
在预设的合理范围内生成参数，如功率限值、能量限值、到达/离开时间等。
'''
U_MAX_BOUND = 1
# 定义一个全局常量 U_MAX_BOUND，值为1。这可能代表功率参数的某个统一缩放因子或默认上限（例如1 kW或1 MW，具体单位取决于上下文）。
U_MIN_BOUND = -1
# 定义一个全局常量 U_MIN_BOUND，值为-1。这可能代表功率参数的某个统一缩放因子或默认下限（例如-1 kW或-1 MW，具体单位取决于上下文）。
X_MAX_BOUND = 10
# 定义一个全局常量 X_MAX_BOUND，值为10。这可能代表能量参数的某个统一缩放因子或默认上限（例如10 kWh或10 MWh，具体单位取决于上下文）。
X_MIN_BOUND = -10
# 定义一个全局常量 X_MIN_BOUND，值为-10。这可能代表能量参数的某个统一缩放因子或默认下限（例如-10 kWh或-10 MWh，具体单位取决于上下文）。
assert U_MAX_BOUND > U_MIN_BOUND
# 断言 U_MAX_BOUND 大于 U_MIN_BOUND，确保功率参数的上下限是合理的。
assert X_MAX_BOUND > X_MIN_BOUND
# 断言 X_MAX_BOUND 大于 X_MIN_BOUND，确保能量参数的上下限是合理的。

def der(T):
    # 定义函数 der(T)，用于为通用的DER (GeneralDER) 生成参数。
    # T: 时间序列的长度。
    u_min = U_MIN_BOUND*np.random.uniform(size=T)
    # 生成最小功率消耗序列 u_min。
    # np.random.uniform(size=T) 会生成一个长度为 T 的数组，其中每个元素都是在 [0, 1) 区间内均匀分布的随机浮点数。
    # 然后乘以 U_MIN_BOUND (-1)，所以 u_min 的元素会在 (-1, 0] 区间内 (假设 U_MIN_BOUND 是负数)。
    u_max = U_MAX_BOUND*np.random.uniform(size=T)  # Can charge up to 2kW
    # 生成最大功率消耗序列 u_max。
    # 元素会在 [0, U_MAX_BOUND) 区间内。
    x_max = X_MAX_BOUND*np.random.uniform(size=T)
    # 生成最大荷电状态序列 x_max。元素会在 [0, X_MAX_BOUND) 区间内。
    x_min = X_MIN_BOUND*np.random.uniform(size=T)
    # 生成最小荷电状态序列 x_min。元素会在 [X_MIN_BOUND, 0) 区间内。
    return u_min, u_max, x_min, x_max

def generation(T):
    rated_power = U_MAX_BOUND*np.random.uniform()  # 5kW rated power
    # 生成一个随机的额定功率，范围在 [0, U_MAX_BOUND) 内。
    # Create sinusoidal generation profile peaking at midday
    # 创建一个正弦发电曲线，峰值在中午。
    t = np.linspace(0, 2 * np.pi, T)
    # 生成一个时间序列 t，从 0 到 2π，长度为 T。
    base_profile = -np.maximum(0, np.random.uniform(0.1) + np.sin(t - np.pi / 2))  # Negative = generation
    # np.sin(t - np.pi / 2) 会生成一个余弦波形（峰值在中间，对应中午）。
    # np.random.uniform(0.1) 增加一个小随机扰动。
    # np.maximum(0, ...) 确保发电量不会是负的（即不会变成耗电）。
    # 最前面的负号表示发电（功率为负）。
    # Scale to realistic power bounds (kW)
    # 生成一个基础发电曲线，峰值在中午。
    # np.maximum(0, np.random.uniform(0.1) + np.sin(t - np.pi / 2)) 生成一个正弦波，峰值在中午。
    # 然后取反，因为发电是负的（表示发电量）。
    # 最后，乘以 rated_power 来缩放发电量。

    # Scale to realistic power bounds (kW)
    u_min = rated_power * base_profile
    # u_min 代表最小功率（即最大发电功率），通过将基础曲线乘以额定功率得到。
    u_max = np.zeros_like(u_min)  # Can curtail to zero but not consume
    # u_max 代表最大功率（即最小发电功率或最大耗电），对于纯发电设备（如PV），通常设为0，表示可以削减发电至0，但不能消耗功率。
    return u_min, u_max


def e1s():
    # 定义函数 e1s()，用于为单向储能系统 (E1S) 生成参数。
    # E1S 通常具有恒定的功率和能量上下限。
    u_max = U_MAX_BOUND*np.random.uniform() 
    # 生成一个随机的最大功率，范围在 [0, U_MAX_BOUND) 内。
    x_max = X_MAX_BOUND*np.random.uniform()
    # 生成一个随机的最大能量，范围在 [0, X_MAX_BOUND) 内。
    x_min = 0
    # x_min 通常为0，表示能量不能为负。
    return u_max, x_min, x_max

def v1g(T):
    # 定义函数 v1g(T)，用于为单向充电电动汽车 (V1G) 生成参数。
    # T: 时间序列的长度。
    u_max = U_MAX_BOUND*np.random.uniform() 
    # 生成一个随机的最大功率，范围在 [0, U_MAX_BOUND) 内。
    a = np.random.randint(T-1)
    # 生成一个随机的到达时间，范围在 [0, T-1) 内。
    d = np.random.randint(a+1,T)
    # 生成一个随机的离开时间，范围在 [a+1, T) 内。

    connected_time = d-a

    e_max = connected_time*u_max*np.random.uniform() 
    # 生成离开时的最大目标荷电状态 e_max。
    # 它被限制在连接时间内以最大功率充电所能达到的总能量乘以一个随机比例（0到1之间），确保 e_max 是可行的。

    e_min = e_max*np.random.uniform()
    # 生成离开时的最小目标荷电状态 e_min。
    # 它被限制在 e_max 的某个比例（0到1之间），确保 e_min 是可行的。
    return a, d, u_max, e_min, e_max

def v2g(T):
    u_min = U_MIN_BOUND*np.random.uniform()
    # 生成一个随机的最小功率，范围在 [U_MIN_BOUND, 0) 内。
    u_max = U_MAX_BOUND*np.random.uniform() 
    # 生成一个随机的最大功率，范围在 [0, U_MAX_BOUND) 内。
    x_max = X_MAX_BOUND*np.random.uniform()
    # 生成一个随机的最大能量，范围在 [0, X_MAX_BOUND) 内。
    x_min = X_MIN_BOUND*np.random.uniform()
    # 生成一个随机的最小能量，范围在 [X_MIN_BOUND, 0) 内。
    # Timing parameters
    a = np.random.randint(T-1)
    # 生成一个随机的到达时间，范围在 [0, T-1) 内。
    d = np.random.randint(a+1,T)
    # 生成一个随机的离开时间，范围在 [a+1, T) 内。
    connected_time = d-a

    e_max = np.minimum(connected_time*u_max*np.random.uniform(), x_max)
    # 生成离开时的最大目标荷电状态 e_max。
    # 它被限制为：1) 连接时间内以最大充电功率充电能达到的能量乘以随机比例；
    # 2) 电池自身的最大容量 x_max。取两者中较小的值。
    e_min = np.random.uniform(0, e_max)
    # 生成离开时的最小目标荷电状态 e_min，范围在 [0, e_max) 内。
    

    return a, d, u_min, u_max, x_min, x_max, e_min, e_max

def e2s():
    # Power parameters (kW)
    u_min = U_MIN_BOUND*np.random.uniform()
    u_max = U_MAX_BOUND*np.random.uniform()  # Can charge up to 2kW
    x_max = X_MAX_BOUND*np.random.uniform()
    x_min = X_MIN_BOUND*np.random.uniform()
    return u_min, u_max, x_min, x_max

def tcl_params(T: int):
    """
    为 Thermostatically Controlled Load (TCL) 生成随机参数。
    这些参数用于初始化TCL论文中描述的模型 x(t) = ax(t-1) + u(t)。

    参数:
        T: int, 时间序列的长度。

    返回:
        tuple: 包含 a_thermal, x0_transformed, u_min_arr, u_max_arr, 
               x_min_phys, x_max_phys
    """

    # 1. 热耗散系数 a_thermal (a)
    # 通常 a 接近 1，表示能量保持性较好。范围 [0, 1)。
    a_thermal = np.random.uniform(0.95, 0.999)

    # 2. 物理状态的上下限 (x_min_physical_state, x_max_physical_state)
    # 这些代表了实际物理量（如温度）允许的波动范围的变换值。
    # 例如，如果x是与设定点的偏差，则这可能对应于死区的一半。
    # 我们随机生成一个“死区宽度”的等效值，然后设定对称的边界。
    deadband_equivalent_width = np.random.uniform(1.0, 5.0) # 任意单位，取决于状态变量的含义
    x_max_physical_state = deadband_equivalent_width / 2.0
    x_min_physical_state = -deadband_equivalent_width / 2.0

    # 3. 变换后的初始状态 x0_transformed (x(0))
    # 初始状态应在物理状态的上下限之间。
    x0_transformed = np.random.uniform(x_min_physical_state, x_max_physical_state)

    # 4. 控制输入 u(t) 的上下限 (u_min_tcl_power, u_max_tcl_power)
    # 这些是变换后的控制输入u(t)的边界，它们通常是恒定的。
    # 假设 u(t) 代表制冷功率输入（负值表示制冷，0表示不制冷）。
    # 或者 u(t) 可以代表一个更一般的控制作用。
    # 我们随机生成一个额定功率等效值。
    # 论文中 u(t) 的推导: u(t) = -(1-a)bp(t)。若 p(t) in [0, p_bar], 则 u(t) in [-(1-a)bp_bar, 0]
    # 这里简化，直接为 u(t) 的上下限采样。
    
    # 假设一个最大功率作用幅度
    max_power_effect = np.random.uniform(0.5, 2.0) # 任意单位，对应 u(t) 的幅度

    # 情况1：主要用于“降低”状态值（例如制冷，u(t) <= 0）
    # u_min_val = -max_power_effect 
    # u_max_val = 0.0
    
    # 情况2：主要用于“提升”状态值（例如制热，u(t) >= 0）
    # u_min_val = 0.0
    # u_max_val = max_power_effect

    # 情况3：允许双向作用或根据具体应用设定
    # 这里我们假设 u(t) 可以是双向的，或者更常见的是单向作用。
    # 为简单起见，假设TCL主要在一个方向上工作（如制冷，u<=0，或制热，u>=0）。
    # 如果是制冷型TCL（功率消耗 p(t) >= 0, u(t) = -(1-a)bp(t) <= 0）:
    effective_u_magnitude = max_power_effect # 代表 (1-a)bp_bar 的大小
    u_min_val = -effective_u_magnitude
    u_max_val = 0.0  # 一般制冷/制热设备不消耗能量来反向操作，而是停止工作

    # 或者，如果是一个可以双向调节的广义u(t)：
    # u_min_val = -max_power_effect / 2
    # u_max_val = max_power_effect / 2
    
    # 创建长度为 T 的恒定值数组
    u_min_arr = np.full(T, u_min_val)
    u_max_arr = np.full(T, u_max_val)

    return a_thermal, x0_transformed, u_min_arr, u_max_arr, \
           x_min_physical_state, x_max_physical_state


