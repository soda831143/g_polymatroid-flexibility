# simulation_utils.py
# 清理版：只保留温度不确定性数据处理函数
# 移除：TCL模拟、地面真实数据生成、参数不确定性相关代码

import numpy as np


def generate_temperature_uncertainty_data(
    omega_combined: np.ndarray,
    use_summer_data: bool = False,
    for_resro: bool = False
):
    """
    为JCC-SRO/Re-SRO算法准备温度预测误差数据
    
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
        
        _print_data_stats("SRO Shape Set", omega_sro_shape)
        _print_data_stats("SRO Calibration Set", omega_sro_calib)
        _print_data_stats("Re-SRO Calibration Set", omega_resro_calib)
        
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
        
        _print_data_stats("Temperature Error Shape Set", omega_shape_set)
        _print_data_stats("Temperature Error Calibration Set", omega_calibration_set)
        
        return omega_shape_set, omega_calibration_set


def generate_ground_truth_data_summer(num_tcls, t_horizon, num_days, use_summer_data=True):
    """
    生成温度预测误差数据 (不再模拟TCL运行)
    
    说明: 
    - 此函数现在只处理温度数据,不进行TCL模拟
    - 仅生成温度预测误差 omega = theta_actual - theta_forecast
    - 保留函数名是为了向后兼容
    
    Args:
        num_tcls: TCL数量 (保留参数但未使用)
        t_horizon: 时间视界
        num_days: 使用的天数
        use_summer_data: 是否使用夏季数据集
        
    Returns:
        omega_combined: 温度预测误差矩阵 (num_days, t_horizon)
    """
    print(f"\n====== 温度预测误差数据生成 ({'夏季数据' if use_summer_data else '合成数据'}) ======")

    if use_summer_data:
        # 加载夏季温度数据
        try:
            all_theta_a_actual = np.load('summer_temps_by_day.npy')[:num_days, :t_horizon]
            print(f"✓ 成功加载夏季温度数据: {all_theta_a_actual.shape}")
            print(f"  温度范围: {all_theta_a_actual.min():.1f}°C 到 {all_theta_a_actual.max():.1f}°C")
        except FileNotFoundError:
            raise FileNotFoundError(
                "找不到夏季温度数据文件 'summer_temps_by_day.npy'\n"
                "请先运行 extract_summer_high_temp_dataset.py 生成数据"
            )
    else:
        raise ValueError("目前仅支持use_summer_data=True，请使用夏季真实数据")
    
    # 生成预测温度 (名义值/确定性预测)
    base_temp = 27.0
    temp_amplitude = 7.0
    time_hours = np.arange(t_horizon)
    theta_a_forecast = base_temp + temp_amplitude * np.cos(2 * np.pi * (time_hours - 15) / 24)
    theta_a_forecast = np.clip(theta_a_forecast, 20.0, 35.0)
    
    print(f"✓ 生成名义预测温度: 范围 [{theta_a_forecast.min():.1f}, {theta_a_forecast.max():.1f}]°C")
    
    # 计算温度预测误差: omega = theta_actual - theta_forecast
    omega_combined = all_theta_a_actual - theta_a_forecast  # (num_days, T)
    
    # 保存温度误差数据
    output_filename = 'summer_all_errors_yesterday.npy'
    np.save(output_filename, omega_combined)
    
    print(f"\n✓ 温度预测误差数据已保存:")
    print(f"  - 文件: {output_filename}")
    print(f"  - 维度: {omega_combined.shape}")
    print(f"  - 误差范围: [{omega_combined.min():.2f}, {omega_combined.max():.2f}]°C")
    print(f"  - 误差均值: {omega_combined.mean():.2f}°C")
    print(f"  - 误差标准差: {omega_combined.std():.2f}°C")
    print("==========================================\n")
    
    return omega_combined


def _print_data_stats(name: str, arr: np.ndarray):
    """打印数据统计信息 (内部辅助函数)"""
    print(f"\n{name} 统计信息:")
    print(f"  - 形状: {arr.shape}")
    print(f"  - 全局范围: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  - 全局均值: {arr.mean():.4f}, 标准差: {arr.std():.4f}")
    if arr.ndim == 2:
        print(f"  - 每时刻最大值: {np.max(arr, axis=0)}")
        print(f"  - 每时刻最小值: {np.min(arr, axis=0)}")
        print(f"  - 每时刻均值: {np.mean(arr, axis=0)}")
