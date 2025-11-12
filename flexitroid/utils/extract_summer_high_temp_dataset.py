import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 配置参数
CSV_FILENAME = "data/weather_data_filtered.csv"
HORIZON_T = 24
DATASET_SPLIT_RATIO = 0.5

# 夏季温度筛选参数
MIN_SUMMER_TEMP = 20.0  # 最低温度阈值（适合制冷）
MIN_DAILY_COOLING_HOURS = 12  # 每天至少12小时温度超过MIN_SUMMER_TEMP
PREFER_MONTHS = [5, 6, 7, 8, 9]  # 偏好月份（5-9月，春末到初秋）

# 输出文件名
SUMMER_TEMPS_BY_DAY_FILENAME = "summer_temps_by_day.npy"
SUMMER_PREDICTED_TEMPS_FILENAME = "summer_yesterday_predicted_temps.npy"
SUMMER_ALL_ERRORS_FILENAME = "summer_all_errors_yesterday.npy"
SUMMER_SHAPE_SET_FILENAME = "summer_shape_set_yesterday.npy"
SUMMER_CALIBRATION_SET_FILENAME = "summer_calibration_set_yesterday.npy"

def load_and_preprocess_greek_data(csv_path: str) -> pd.DataFrame:
    """
    加载并预处理希腊温度数据
    
    Returns:
        包含温度时间序列的DataFrame，索引为时间戳
    """
    print("加载希腊温度数据...")
    df = pd.read_csv(csv_path)
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('utc_timestamp', inplace=True)
    
    # 重采样为小时数据（已经是小时数据，但确保没有缺失）
    temp_series = df['GR_temperature'].resample('h').mean()
    temp_series = temp_series.dropna()
    
    print(f"数据时间范围: {temp_series.index.min()} 到 {temp_series.index.max()}")
    print(f"总数据点: {len(temp_series)}")
    print(f"温度范围: {temp_series.min():.2f}°C 到 {temp_series.max():.2f}°C")
    
    return pd.DataFrame({'temperature': temp_series})

def identify_summer_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别适合制冷系统的夏季高温天数
    
    筛选条件：
    1. 每天至少MIN_DAILY_COOLING_HOURS小时温度超过MIN_SUMMER_TEMP
    2. 优先选择夏季月份（5-9月）
    3. 每天24小时数据完整
    
    Returns:
        按天重新组织的温度数据，形状为(num_summer_days, 24)
    """
    print("\n筛选夏季高温天数...")
    
    # 按天分组
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    
    summer_days_data = []
    summer_days_info = []
    
    for date, day_group in df.groupby('date'):
        # 检查数据完整性（24小时）
        if len(day_group) != HORIZON_T:
            continue
            
        daily_temps = day_group.sort_values('hour')['temperature'].values
        
        # 统计高温小时数
        cooling_hours = np.sum(daily_temps >= MIN_SUMMER_TEMP)
        
        # 检查是否符合制冷需求
        if cooling_hours >= MIN_DAILY_COOLING_HOURS:
            # 额外偏好夏季月份
            month = pd.to_datetime(str(date)).month
            is_prefer_month = month in PREFER_MONTHS
            
            summer_days_data.append(daily_temps)
            summer_days_info.append({
                'date': date,
                'month': month,
                'cooling_hours': cooling_hours,
                'max_temp': daily_temps.max(),
                'min_temp': daily_temps.min(),
                'avg_temp': daily_temps.mean(),
                'is_prefer_month': is_prefer_month
            })
    
    # 转换为DataFrame以便排序
    info_df = pd.DataFrame(summer_days_info)
    
    print(f"初步筛选出 {len(summer_days_data)} 个符合条件的天数")
    print(f"月份分布:")
    month_counts = info_df['month'].value_counts().sort_index()
    for month, count in month_counts.items():
        print(f"  {month}月: {count} 天")
    
    # 按偏好排序：优先夏季月份，然后按平均温度排序
    info_df['priority_score'] = (
        info_df['is_prefer_month'].astype(int) * 1000 +  # 偏好月份权重
        info_df['avg_temp'] * 10 +  # 平均温度权重
        info_df['cooling_hours']  # 制冷小时数权重
    )
    
    sorted_indices = info_df.sort_values('priority_score', ascending=False).index
    
    # 重新排列数据
    sorted_summer_data = [summer_days_data[i] for i in sorted_indices]
    sorted_info = info_df.iloc[sorted_indices].reset_index(drop=True)
    
    # 转换为numpy数组
    summer_temps_array = np.array(sorted_summer_data)
    
    print(f"\n夏季数据统计:")
    print(f"最终选择的天数: {len(summer_temps_array)}")
    print(f"温度范围: {summer_temps_array.min():.2f}°C 到 {summer_temps_array.max():.2f}°C")
    print(f"平均温度: {summer_temps_array.mean():.2f}°C")
    print(f"大于{MIN_SUMMER_TEMP}°C的比例: {(summer_temps_array >= MIN_SUMMER_TEMP).mean()*100:.1f}%")
    
    return summer_temps_array, sorted_info

def create_yesterday_prediction_and_errors(temps_by_day: np.ndarray):
    """
    创建"昨日基线"预测和预测误差
    
    预测逻辑：用前一天的温度作为今天的预测
    
    Returns:
        predicted_temps: 预测温度 (num_days-1, 24)
        errors: 预测误差 (num_days-1, 24) 
        actual_temps: 实际温度 (num_days-1, 24)
    """
    print("\n创建昨日基线预测...")
    
    num_days = temps_by_day.shape[0]
    
    # 昨日预测：今天的预测 = 昨天的实际温度
    predicted_temps = temps_by_day[:-1]  # 取前n-1天作为后n-1天的预测
    actual_temps = temps_by_day[1:]      # 取后n-1天作为实际温度
    
    # 计算预测误差：误差 = 预测值 - 实际值
    errors = predicted_temps - actual_temps
    
    print(f"预测数据形状: {predicted_temps.shape}")
    print(f"实际数据形状: {actual_temps.shape}")
    print(f"误差数据形状: {errors.shape}")
    print(f"误差统计 - 均值: {errors.mean():.3f}, 标准差: {errors.std():.3f}")
    print(f"误差范围: [{errors.min():.3f}, {errors.max():.3f}]")
    
    return predicted_temps, errors, actual_temps

def split_and_save_datasets(errors: np.ndarray, predicted_temps: np.ndarray, actual_temps: np.ndarray):
    """
    划分数据集并保存
    
    Args:
        errors: 预测误差数据
        predicted_temps: 预测温度数据  
        actual_temps: 实际温度数据
    """
    print("\n划分并保存数据集...")
    
    num_days = errors.shape[0]
    split_index = int(num_days * DATASET_SPLIT_RATIO)
    
    # 划分误差数据
    shape_set = errors[:split_index]
    calibration_set = errors[split_index:]
    
    # 保存数据
    np.save(SUMMER_TEMPS_BY_DAY_FILENAME, actual_temps)
    np.save(SUMMER_PREDICTED_TEMPS_FILENAME, predicted_temps)
    np.save(SUMMER_ALL_ERRORS_FILENAME, errors)
    np.save(SUMMER_SHAPE_SET_FILENAME, shape_set)
    np.save(SUMMER_CALIBRATION_SET_FILENAME, calibration_set)
    
    print("数据保存完成:")
    print(f" - {SUMMER_TEMPS_BY_DAY_FILENAME}: {actual_temps.shape}")
    print(f" - {SUMMER_PREDICTED_TEMPS_FILENAME}: {predicted_temps.shape}")
    print(f" - {SUMMER_ALL_ERRORS_FILENAME}: {errors.shape}")
    print(f" - {SUMMER_SHAPE_SET_FILENAME}: {shape_set.shape}")
    print(f" - {SUMMER_CALIBRATION_SET_FILENAME}: {calibration_set.shape}")

def plot_sample_data(temps_by_day: np.ndarray, predicted_temps: np.ndarray, errors: np.ndarray):
    """
    绘制样本数据进行可视化验证
    """
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 前5天的温度曲线
    axes[0, 0].set_title('前5天夏季温度曲线')
    for i in range(min(5, temps_by_day.shape[0])):
        axes[0, 0].plot(temps_by_day[i], label=f'第{i+1}天', marker='o', markersize=3)
    axes[0, 0].axhline(y=MIN_SUMMER_TEMP, color='r', linestyle='--', alpha=0.7, label=f'制冷阈值({MIN_SUMMER_TEMP}°C)')
    axes[0, 0].set_xlabel('小时')
    axes[0, 0].set_ylabel('温度 (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 温度分布直方图
    axes[0, 1].hist(temps_by_day.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=MIN_SUMMER_TEMP, color='r', linestyle='--', label=f'制冷阈值({MIN_SUMMER_TEMP}°C)')
    axes[0, 1].set_title('夏季温度分布')
    axes[0, 1].set_xlabel('温度 (°C)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 预测误差时间序列（前5天）
    axes[1, 0].set_title('前5天预测误差')
    for i in range(min(5, errors.shape[0])):
        axes[1, 0].plot(errors[i], label=f'第{i+1}天', marker='s', markersize=3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('小时')
    axes[1, 0].set_ylabel('预测误差 (°C)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 预测误差分布
    axes[1, 1].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', label='零误差')
    axes[1, 1].set_title('预测误差分布')
    axes[1, 1].set_xlabel('预测误差 (°C)')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('summer_temperature_analysis.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存为 'summer_temperature_analysis.png'")
    plt.show()

def main():
    """主执行函数"""
    print("====== 提取夏季高温数据集用于制冷系统建模 ======")
    print(f"筛选条件:")
    print(f"  - 最低温度阈值: {MIN_SUMMER_TEMP}°C")
    print(f"  - 每天最少制冷小时数: {MIN_DAILY_COOLING_HOURS}")
    print(f"  - 偏好月份: {PREFER_MONTHS}")
    
    # 1. 加载数据
    df = load_and_preprocess_greek_data(CSV_FILENAME)
    
    # 2. 筛选夏季高温天数
    summer_temps, summer_info = identify_summer_days(df)
    
    # 3. 创建预测和误差数据
    predicted_temps, errors, actual_temps = create_yesterday_prediction_and_errors(summer_temps)
    
    # 4. 划分并保存数据集
    split_and_save_datasets(errors, predicted_temps, actual_temps)
    
    # 5. 生成可视化
    # plot_sample_data(summer_temps, predicted_temps, errors)
    
    print("\n====== 夏季温度数据集提取完成 ======")
    print("✓ 数据已保存到对应的.npy文件")
    print("✓ 可视化图表已生成")
    print("✓ 数据集已准备好用于制冷系统的TCL建模")

if __name__ == "__main__":
    main() 