import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- 配置参数 ---
# 这是从 generate_temperature_error_dataset.py 脚本中复制的参数，必须保持一致
CSV_FILENAME = "jena_climate_2009_2016.csv"  # 修改为正确的文件名
SEQUENCE_LENGTH = 72
HORIZON_T = 24
ERROR_DATA_FILENAME = "temperature_error_trajectories.npy"

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    加载CSV文件，进行预处理，并返回小时级的温度时间序列DataFrame。
    与训练脚本保持一致。
    """
    print("加载并预处理数据...")
    df = pd.read_csv(CSV_FILENAME)
    # 将 'Date Time' 列转换为datetime对象并设为索引
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df.set_index('Date Time', inplace=True)

    # 选取温度列，并重采样为小时数据（取平均值）
    temp_series = df['T (degC)'].resample('h').mean()  # 使用 'h' 而不是 'H'
    temp_series.dropna(inplace=True) # 去除可能因重采样产生的缺失值
    
    # 转换为DataFrame格式
    temp_df = pd.DataFrame({'T (degC)': temp_series})
    print("数据预处理完成。")
    return temp_df

def calculate_metrics(y_true, y_pred):
    """计算并打印核心评估指标。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算 MAPE (Mean Absolute Percentage Error)，注意处理真实值为0的情况
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    print("\n--- 量化评估指标 ---")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    print("--------------------")

def plot_diagnostics(y_true, y_pred, errors):
    """绘制一系列诊断图表来评估模型性能。"""
    print("\n正在生成诊断图表...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)

    # --- 子图1: 预测值 vs. 真实值对比图 (抽样展示) ---
    ax1 = fig.add_subplot(gs[0, :])
    sample_size = min(500, len(y_true)) # 只画前500个点，避免图像过于拥挤
    ax1.plot(y_true[:sample_size], label='真实值 (Actual)', color='royalblue', marker='.', linestyle='-')
    ax1.plot(y_pred[:sample_size], label='预测值 (Predicted)', color='crimson', marker='.', linestyle='--')
    ax1.set_title(f'真实值 vs. 预测值 (前 {sample_size} 个点)', fontsize=16)
    ax1.set_xlabel('时间步 (Time Step)')
    ax1.set_ylabel('温度 (°C)')
    ax1.legend()
    ax1.grid(True)

    # --- 子图2: 误差自相关图 (ACF) ---
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(errors, lags=48, ax=ax2, title='误差自相关图 (ACF Plot of Errors)')
    ax2.set_xlabel('延迟 (Lag)')
    ax2.set_ylabel('自相关系数')
    ax2.grid(True)

    # --- 子图3: 预测值 vs. 真实值散点图 ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(y_true, y_pred, alpha=0.3, color='darkorange', edgecolors='w')
    # 绘制理想情况下的对角线 (y=x)
    lims = [
        np.min([ax3.get_xlim(), ax3.get_ylim()]),
        np.max([ax3.get_xlim(), ax3.get_ylim()]),
    ]
    ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='理想情况 (y=x)')
    ax3.set_title('预测值 vs. 真实值 散点图', fontsize=16)
    ax3.set_xlabel('真实值 (Actual Values)')
    ax3.set_ylabel('预测值 (Predicted Values)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('lstm_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主执行函数。"""
    # 1. 加载误差数据
    print(f"正在加载误差数据: '{ERROR_DATA_FILENAME}'")
    try:
        errors_matrix = np.load(ERROR_DATA_FILENAME)
        print(f"误差数据形状: {errors_matrix.shape}")
        
        # 将误差轨迹重构为连续的误差序列
        # 方法：取每条轨迹的第一个元素，然后从最后一条轨迹开始取剩余元素
        errors_flat = errors_matrix[:, 0]  # 所有轨迹的第一个元素
        for i in range(1, errors_matrix.shape[1]):
            errors_flat = np.append(errors_flat, errors_matrix[-1, i])
        
        print(f"重构后的误差序列长度: {len(errors_flat)}")
        
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{ERROR_DATA_FILENAME}'。请先运行 `generate_temperature_error_dataset.py`。")
        return

    # 2. 加载原始数据并进行与训练时相同的预处理
    print(f"正在加载原始数据: '{CSV_FILENAME}'")
    df = load_and_preprocess_data(CSV_FILENAME)

    # 3. 划分训练集和验证集 (与训练时保持一致)
    split_fraction = 0.715
    train_split = int(split_fraction * int(df.shape[0]))
    df_val = df.loc[df.index[train_split] :]
    
    print(f"验证集数据点数量: {len(df_val)}")

    # 4. 对齐数据
    # 真实值：从验证集中取对应长度的数据
    y_true = df_val['T (degC)'].values[SEQUENCE_LENGTH : SEQUENCE_LENGTH + len(errors_flat)]
    
    # 从误差和真实值反推出预测值
    y_pred = y_true + errors_flat
    
    print(f"真实值长度: {len(y_true)}")
    print(f"预测值长度: {len(y_pred)}")
    print(f"误差长度: {len(errors_flat)}")
    
    # 5. 计算量化指标
    calculate_metrics(y_true, y_pred)

    # 6. 绘制诊断图
    plot_diagnostics(y_true, y_pred, errors_flat)
    
    # 7. 保存评估结果
    results_df = pd.DataFrame({
        'True_Temperature': y_true,
        'Predicted_Temperature': y_pred,
        'Error': errors_flat
    })
    results_df.to_csv('lstm_evaluation_results.csv', index=False)
    print("\n评估结果已保存到 'lstm_evaluation_results.csv'")

if __name__ == "__main__":
    main()