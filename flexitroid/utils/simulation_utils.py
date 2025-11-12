# simulation_utils.py

import numpy as np
import pandas as pd
from flexitroid.devices.tcl import TCL

def create_true_tcl_fleet(num_tcls: int, t_horizon: int, true_params_dist: dict, theta_a_actual: np.ndarray, build_g_poly: bool = True) -> list:
    """
    (研究者"上帝视角"或"聚合商视角") 创建一个异构TCL群体。
    
    仅考虑制冷场景：所有TCL仅能制冷，不能制热。
    当 build_g_poly=False 时（上帝视角）: 只创建具有物理参数的TCL对象，用于模拟。
    当 build_g_poly=True 时（聚合商视角）: 创建TCL对象并构建其g-polymatroid模型，用于灵活性聚合。
    """
    tcl_fleet = []
    print(f"--- 正在创建 {num_tcls} 个TCL群体 (模式: {'模型构建' if build_g_poly else '物理模拟'}) ---")
    
    attempts = 0
    while len(tcl_fleet) < num_tcls and attempts < num_tcls * 10: # 限制尝试次数
        attempts += 1
        
        # 1. 从分布中采样物理参数 (所有模式下都需要)
        R_th = np.random.uniform(*true_params_dist['R_th_range'])
        C_th = np.random.uniform(*true_params_dist['C_th_range'])
        P_m = np.random.uniform(*true_params_dist['P_m_range'])
        eta = np.random.uniform(*true_params_dist['eta_range'])
        theta_r = np.random.uniform(*true_params_dist['theta_r_range'])
        delta_val = np.random.uniform(*true_params_dist['delta_val_range'])
        
        a = 1-1/(R_th*C_th)
        b = R_th * eta
        delta = 1
        
        params = {
            'T': t_horizon, 'a': a, 'b': b, 'C_th': C_th, 'eta': eta, 'delta': delta,
            'P_m': P_m, # 仅最大制冷功率
            'theta_r': theta_r, 'x0': 0.0,
            'delta_val': delta_val
        }
        # 注意：auto_adjust_tcl_params 可能需要根据情况调整或移除，取决于其内部逻辑
        # params = auto_adjust_tcl_params(params, theta_a_forecast=theta_a_actual)

        # 2. 根据模式创建TCL对象
        if not build_g_poly:
            # 【物理模拟模式】: 不需要任何可行性检查，直接创建对象
            # 任何一组参数都代表一个可能的物理实体，无论它性能好坏
            tcl_instance = TCL(params, build_g_poly=False, theta_a_forecast=theta_a_actual)
            tcl_fleet.append(tcl_instance)
        else:
            # 【模型构建模式】: 需要进行严格的数学可行性检查
            try:
                # 这会触发内部的g-polymatroid计算
                tcl_instance = TCL(params, build_g_poly=True, theta_a_forecast=theta_a_actual)
                
                # 检查最关键的约束：整个时间跨度内的总能量
                total_A = frozenset(range(t_horizon))
                if tcl_instance.p(total_A) > tcl_instance.b(total_A):
                    # 如果 p > b，说明该TCL模型在数学上不可行，丢弃
                    continue
                
                tcl_fleet.append(tcl_instance)
            except ValueError:
                # TCL.__init__ 内部如果发现不可行，可能会抛出ValueError
                continue

    if len(tcl_fleet) < num_tcls:
        raise RuntimeError(f"在 {attempts} 次尝试后，未能创建足够数量的 {'可行TCL模型' if build_g_poly else 'TCL实例'}！")
        
    print(f"TCL群体创建完成，成功创建 {len(tcl_fleet)} 个。")
    return tcl_fleet

# simulation_utils.py -> simulate_autonomous_tcl_operation

def simulate_autonomous_tcl_operation(tcl: TCL, theta_a_actual: np.ndarray) -> np.ndarray:
    """
    (研究者“上帝视角”) 模拟单个【持续调节型】TCL在无调度干预下的自主运行。
    
    物理逻辑 (混合控制策略):
    - 当室内温度 θ(k) > θᵣ + δ (超过舒适上界) 时，开启最大功率制冷 P(k) = P_m。
    - 当 θᵣ - δ ≤ θ(k) ≤ θᵣ + δ (在舒适区间内) 时，输出恰好抵消当前热负荷的“维持功率”。
    - 当 θ(k) < θᵣ - δ (低于舒适下界) 时，设备为仅制冷，功率为0。
    """
    T = tcl.T
    P_hist = np.zeros(T)
    
    theta_r, delta_val = tcl.theta_r, tcl.delta_val
    a, b = tcl.a, tcl.b_coef
    
    P_max_cool = tcl.P_m 
    
    theta_indoor = np.zeros(T + 1)
    theta_indoor[0] = theta_r
    
    for k in range(T):
        upper_bound = theta_r + delta_val  # 舒适温度上界
        lower_bound = theta_r - delta_val  # 舒适温度下界

        # --- 【核心修改点：持续调节型TCL的混合控制逻辑】 ---
        
        if theta_indoor[k] > upper_bound:
            # 1. 温度过高：使用最大功率全力制冷
            P_hist[k] = P_max_cool
            
        elif theta_indoor[k] < lower_bound:
            # 2. 温度过低：仅制冷设备，关闭
            P_hist[k] = 0.0
            
        else:
            # 3. 温度在舒适区间内：计算并输出“维持功率”
            # 这个功率刚好可以抵消当前来自外界的热量，使得温度趋于稳定。
            # 其计算公式源于热力学方程的稳态解。
            maintenance_power = (theta_a_actual[k] - theta_indoor[k]) / b
            
            # 约束维持功率：
            # a. 只能制冷，功率不能为负。
            # b. 不能超过设备的最大功率。
            P_hist[k] = np.clip(maintenance_power, 0, P_max_cool)
            
        # --- 【修改结束】 ---
            
        # 热力学方程保持不变
        theta_indoor[k+1] = a * theta_indoor[k] + (1 - a) * (theta_a_actual[k] - b * P_hist[k])

    return P_hist

# def generate_ground_truth_data(num_tcls, t_horizon, num_days):
#     print("\n====== Phase 1: 开始生成地面真实数据 ======")

#     true_params_dist = {
#         'R_th_range': (2.0, 5.0), 'C_th_range': (20, 30),
#         'P_m_range': (60.0, 80.0), 'eta_range': (2.5, 3.5),
#         'theta_r_range': (22.0, 26.0), 'delta_val_range': (3.0, 4.0)
#     }

#     all_theta_a_actual = np.load('temps_by_day.npy')[:num_days, :t_horizon]

#     base_temp = 27.0
#     temp_amplitude = 6.0
#     T_HORIZON = 24
#     time_hours = np.arange(T_HORIZON)
#     theta_a_forecast = base_temp + temp_amplitude * np.cos(2 * np.pi * (time_hours - 15) / 24)
#     theta_a_forecast = np.clip(theta_a_forecast, 20.0, 35.0)
    
#     # 时间从0到23小时，凌晨3点最低，下午3点最高
#     tcl_fleet = create_true_tcl_fleet(num_tcls, t_horizon, true_params_dist, theta_a_forecast, build_g_poly=False)

#     all_P_agg_hist = []
#     for day_idx in range(num_days):
#         P_agg_daily = np.zeros(t_horizon)
#         for tcl in tcl_fleet:
#             P_i_daily = simulate_autonomous_tcl_operation(tcl, all_theta_a_actual[day_idx])
#             P_agg_daily += P_i_daily
#         all_P_agg_hist.append(P_agg_daily)

#     all_P_agg_hist = np.array(all_P_agg_hist)
#     np.save('P_agg_hist.npy', all_P_agg_hist)
#     print("\n地面真实数据生成完毕并已保存：")
#     print(f" - P_agg_hist.npy (维度: {all_P_agg_hist.shape})")
#     print("==========================================\n")

def compute_baseline_power_aggregate(tcl_fleet: list, theta_a_forecast: np.ndarray) -> np.ndarray:
    """
    (聚合商视角) 计算给定TCL群体在预测温度下的基线聚合功率。
    
    物理模型：P₀(k) = (θₐ - θᵣ)/b
    - 当 θₐ > θᵣ 时，P₀ > 0 (理论上需要制冷)
    - 当 θₐ < θᵣ 时，P₀ < 0 (理论上需要制热，但TCL实际不能制热)
    
    注意：基线功率P₀可以为负，这是理论参考值。
    实际功率约束P(k) ≥ 0在其他地方处理。
    
    Args:
        tcl_fleet: TCL群体列表
        theta_a_forecast: 预测的室外温度序列
        
    Returns:
        基线聚合功率序列 P0_agg
    """
    T = len(theta_a_forecast)
    P0_agg = np.zeros(T)
    
    for tcl in tcl_fleet:
        # 基线功率公式：P₀(k) = (θₐ - θᵣ)/b
        # 保持原始值，不进行截断
        P0_i_unconstrained = (theta_a_forecast - tcl.theta_r) / tcl.b_coef
        P0_i = np.maximum(0, P0_i_unconstrained)
        P0_agg += P0_i
    
    return P0_agg

def generate_temperature_uncertainty_data(
    omega_combined: np.ndarray,
    use_summer_data: bool = False,
    for_resro: bool = False
):
    """
    (聚合商视角) 为JCC-SRO/Re-SRO算法准备温度预测误差数据。
    
    数据划分策略:
    - JCC-SRO (for_resro=False): 全部数据D划分为 1/2形状集 + 1/2校准集
    - JCC-Re-SRO (for_resro=True): 全部数据D划分为:
        * 1/4 SRO形状集
        * 1/4 SRO校准集  
        * 1/2 Re-SRO校准集
    
    Args:
        omega_combined: 温度预测误差数据 (num_days, T)
        use_summer_data: 是否使用夏季数据标识 (仅用于文件命名)
        for_resro: 是否为Re-SRO算法准备三份数据
        
    Returns:
        如果for_resro=False: (omega_shape_set, omega_calibration_set)
        如果for_resro=True: (omega_sro_shape, omega_sro_calib, omega_resro_calib)
    """
    data_type = "夏季" if use_summer_data else "常规"
    algo_type = "Re-SRO(三份)" if for_resro else "SRO(两份)"
    print(f"\n====== 温度预测误差数据准备 ({data_type}数据, {algo_type}) ======")
    print(f"输入温度误差数据: {omega_combined.shape}")
    
    num_samples = omega_combined.shape[0]
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    
    file_suffix = "_summer" if use_summer_data else ""
    
    if for_resro:
        # Re-SRO: 三份数据 (1/4 + 1/4 + 1/2)
        split1 = num_samples // 4
        split2 = num_samples // 2
        
        sro_shape_idx = idx[:split1]
        sro_calib_idx = idx[split1:split2]
        resro_calib_idx = idx[split2:]
        
        omega_sro_shape = omega_combined[sro_shape_idx]
        omega_sro_calib = omega_combined[sro_calib_idx]
        omega_resro_calib = omega_combined[resro_calib_idx]
        
        # 保存文件
        sro_shape_file = f'omega_sro_shape{file_suffix}.npy'
        sro_calib_file = f'omega_sro_calib{file_suffix}.npy'
        resro_calib_file = f'omega_resro_calib{file_suffix}.npy'
        
        np.save(sro_shape_file, omega_sro_shape)
        np.save(sro_calib_file, omega_sro_calib)
        np.save(resro_calib_file, omega_resro_calib)
        
        print(f"  Re-SRO数据划分:")
        print(f"    - SRO形状集 (1/4): {omega_sro_shape.shape} → {sro_shape_file}")
        print(f"    - SRO校准集 (1/4): {omega_sro_calib.shape} → {sro_calib_file}")
        print(f"    - Re-SRO校准集 (1/2): {omega_resro_calib.shape} → {resro_calib_file}")
        print("==========================================\n")
        
        print_shape_set_stats("SRO Shape Set", omega_sro_shape)
        print_shape_set_stats("SRO Calibration Set", omega_sro_calib)
        print_shape_set_stats("Re-SRO Calibration Set", omega_resro_calib)
        
        return omega_sro_shape, omega_sro_calib, omega_resro_calib
    
    else:
        # SRO: 两份数据 (1/2 + 1/2)
        split_idx = num_samples // 2
        shape_idx = idx[:split_idx]
        calib_idx = idx[split_idx:]
        
        omega_shape_set = omega_combined[shape_idx]
        omega_calibration_set = omega_combined[calib_idx]
        
        # 保存文件
        shape_file = f'omega_shape_set{file_suffix}.npy'
        calib_file = f'omega_calibration_set{file_suffix}.npy'
        
        np.save(shape_file, omega_shape_set)
        np.save(calib_file, omega_calibration_set)
        
        print(f"  SRO数据划分:")
        print(f"    - 形状集 (1/2): {omega_shape_set.shape} → {shape_file}")
        print(f"    - 校准集 (1/2): {omega_calibration_set.shape} → {calib_file}")
        print("==========================================\n")
        
        print_shape_set_stats("Temperature Error Shape Set", omega_shape_set)
        print_shape_set_stats("Temperature Error Calibration Set", omega_calibration_set)
        
        return omega_shape_set, omega_calibration_set


def generate_ground_truth_data_summer(num_tcls, t_horizon, num_days, use_summer_data=True):
    """
    使用夏季高温数据生成地面真实数据
    
    Args:
        num_tcls: TCL数量
        t_horizon: 时间视界
        num_days: 使用的天数
        use_summer_data: 是否使用夏季数据集，如果False则使用原始合成数据
    """
    print(f"\n====== Phase 1: 开始生成地面真实数据 ({'夏季数据' if use_summer_data else '合成数据'}) ======")

    true_params_dist = {
        'R_th_range': (2.0, 3.0), 'C_th_range': (10.0, 15.0),
        'P_m_range': (10.0, 20.0), 'eta_range': (2.5, 4.0),
        'theta_r_range': (22.0, 24.0), 'delta_val_range': (1.0, 2.0)
    }

    if use_summer_data:
        # 使用夏季数据集
        try:
            all_theta_a_actual = np.load('summer_temps_by_day.npy')[:num_days, :t_horizon]
            print(f"成功加载夏季温度数据: {all_theta_a_actual.shape}")
            print(f"夏季温度范围: {all_theta_a_actual.min():.1f}°C 到 {all_theta_a_actual.max():.1f}°C")
        except FileNotFoundError:
            print("错误: 找不到夏季温度数据文件，请先运行 extract_summer_high_temp_dataset.py")
            print("将回退到使用合成温度数据...")
            use_summer_data = False
    
    # if not use_summer_data:
    #     # 使用原始合成数据
    #     try:
    #         all_theta_a_actual = np.load('temps_by_day.npy')[:num_days, :t_horizon]
    #         print(f"使用原始温度数据: {all_theta_a_actual.shape}")
    #     except FileNotFoundError:
    #         print("生成合成温度数据...")
    #         # 生成合成温度数据
    #         base_temp = 27.0
    #         temp_amplitude = 6.0
    #         time_hours = np.arange(t_horizon)
    #         theta_a_forecast = base_temp + temp_amplitude * np.cos(2 * np.pi * (time_hours - 15) / 24)
    #         theta_a_forecast = np.clip(theta_a_forecast, 20.0, 35.0)
    #         all_theta_a_actual = np.tile(theta_a_forecast, (num_days, 1))

    # 创建TCL群体（用于温度预测的基准温度）
    base_temp = 27.0
    temp_amplitude = 7.0
    time_hours = np.arange(t_horizon)
    theta_a_forecast = base_temp + temp_amplitude * np.cos(2 * np.pi * (time_hours - 15) / 24)
    theta_a_forecast = np.clip(theta_a_forecast, 20.0, 35.0)    

    tcl_fleet = create_true_tcl_fleet(num_tcls, t_horizon, true_params_dist, theta_a_forecast, build_g_poly=False)

    all_P_agg_hist = []
    for day_idx in range(num_days):
        P_agg_daily = np.zeros(t_horizon)
        for tcl in tcl_fleet:
            P_i_daily = simulate_autonomous_tcl_operation(tcl, all_theta_a_actual[day_idx])
            P_agg_daily += P_i_daily
        all_P_agg_hist.append(P_agg_daily)

    all_P_agg_hist = np.array(all_P_agg_hist)
    
    # 根据数据类型保存到不同文件
    output_filename = 'P_agg_hist_summer.npy' if use_summer_data else 'P_agg_hist.npy'
    np.save(output_filename, all_P_agg_hist)
    
    print(f"\n地面真实数据生成完毕并已保存：")
    print(f" - {output_filename} (维度: {all_P_agg_hist.shape})")
    print(f" - 功率范围: {all_P_agg_hist.min():.1f} 到 {all_P_agg_hist.max():.1f} kW")
    print("==========================================\n")

def verify_cooling_only_model_consistency(tcl_params: dict, theta_a_forecast: np.ndarray) -> bool:
    """
    验证仅制冷模式下的数学模型一致性。
    
    检查项目：
    1. 基线功率计算：P₀(k) = (θₐ - θᵣ)/b
    2. 控制变量定义：u(k) = P(k) - P₀(k)  
    3. 功率约束：P(k) ≥ 0 且 P(k) ≤ P_m
    4. 控制变量边界：u_min = -P₀(k), u_max = P_m - P₀(k)
    
    Returns:
        bool: 如果所有检查通过返回True，否则返回False
    """
    print("\n====== 验证仅制冷模式数学模型一致性 ======")
    
    # 提取参数
    theta_r = tcl_params['theta_r']
    b = tcl_params['b']
    P_m = tcl_params['P_m']
    
    # 1. 检查基线功率计算
    # P0_forecast = (theta_a_forecast - theta_r) / b
    P0_unconstrained = (theta_a_forecast - theta_r) / b
    P0_forecast = np.maximum(0, P0_unconstrained)
    print(f"基线功率 P₀ 范围: [{P0_forecast.min():.2f}, {P0_forecast.max():.2f}]")
    
    # 2. 检查控制变量边界
    u_min = -P0_forecast  # 来自 P(k) ≥ 0
    u_max = P_m - P0_forecast  # 来自 P(k) ≤ P_m
    
    print(f"控制变量 u_min 范围: [{u_min.min():.2f}, {u_min.max():.2f}]")
    print(f"控制变量 u_max 范围: [{u_max.min():.2f}, {u_max.max():.2f}]")
    
    # 3. 验证约束一致性
    consistency_checks = []
    
    # 检查 u_min ≤ u_max 对所有时刻成立
    check1 = np.all(u_min <= u_max)
    consistency_checks.append(("u_min ≤ u_max", check1))
    
    # 检查当P₀ < 0时，u_min > 0 (因为u_min = -P₀)
    negative_P0_times = P0_forecast < 0
    if np.any(negative_P0_times):
        check2 = np.all(u_min[negative_P0_times] > 0)
        consistency_checks.append(("P₀ < 0 时 u_min > 0", check2))
    
    # 检查当P₀ > P_m时，u_max < 0 (物理不可行的情况)
    excessive_P0_times = P0_forecast > P_m
    if np.any(excessive_P0_times):
        check3 = np.all(u_max[excessive_P0_times] < 0)
        consistency_checks.append(("P₀ > P_m 时 u_max < 0 (不可行)", check3))
    
    # 4. 打印检查结果
    all_passed = True
    for check_name, result in consistency_checks:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    # 5. 物理意义验证
    print("\n物理意义验证:")
    print(f"- 当外温 > 设定温度时，P₀ > 0 (需要制冷)")
    print(f"- 当外温 < 设定温度时，P₀ < 0 (理论需制热，但TCL不能制热)")
    print(f"- 控制变量 u(k) 允许在基线功率基础上进行灵活性调度")
    print(f"- 实际功率 P(k) = P₀(k) + u(k) 必须满足 0 ≤ P(k) ≤ P_m")
    
    if all_passed:
        print("\n✓ 数学模型一致性验证通过！")
    else:
        print("\n✗ 数学模型一致性验证失败，请检查参数设置！")
    
    print("==========================================\n")
    return all_passed

def print_shape_set_stats(name, arr):
    print(f"\n{name} 统计信息:")
    print(f"  - shape: {arr.shape}")
    print(f"  - 全局最大值: {arr.max():.4f}")
    print(f"  - 全局最小值: {arr.min():.4f}")
    print(f"  - 每维最大值: {np.max(arr, axis=0)}")
    print(f"  - 每维最小值: {np.min(arr, axis=0)}")
    print(f"  - 每维均值: {np.mean(arr, axis=0)}")
    print(f"  - 每维标准差: {np.std(arr, axis=0)}")