# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import zipfile
import requests
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# --- 常量定义 ---
CSV_FILENAME = "jena_climate_2009_2016.csv"

# 研究参数
HORIZON_T = 24  # 我们研究的时间分辨率为每小时，一天24个时间点
DATASET_SPLIT_RATIO = 0.5    # 形状学习集(D1)与尺寸校准集(D2)的划分比例

# 输出文件名
OUTPUT_FILENAME = "temperature_error_trajectories.npy"
SHAPE_SET_FILENAME = "shape_set.npy"
CALIBRATION_SET_FILENAME = "calibration_set.npy"

# LSTM模型和数据序列化参数 - 改进版本
SEQUENCE_LENGTH = 168  # 使用过去168小时（7天）的数据作为输入序列
BATCH_SIZE = 128  # 减小批次大小以提高稳定性
EPOCHS = 50  # 增加训练周期

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    加载CSV文件，进行预处理，并返回小时级的温度时间序列DataFrame。
    增加更多特征以提高预测精度。
    """
    print("加载并预处理数据...")
    df = pd.read_csv(CSV_FILENAME)
    # 将 'Date Time' 列转换为datetime对象并设为索引
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df.set_index('Date Time', inplace=True)

    # 选取多个相关特征列
    feature_columns = ['T (degC)', 'p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']
    
    # 重采样为小时数据（取平均值）
    hourly_df = df[feature_columns].resample('h').mean()
    hourly_df.dropna(inplace=True)
    
    # 添加时间特征
    hourly_df['hour'] = hourly_df.index.hour
    hourly_df['day_of_week'] = hourly_df.index.dayofweek
    hourly_df['month'] = hourly_df.index.month
    hourly_df['day_of_year'] = hourly_df.index.dayofyear
    
    # 添加周期性特征
    hourly_df['hour_sin'] = np.sin(2 * np.pi * hourly_df['hour'] / 24)
    hourly_df['hour_cos'] = np.cos(2 * np.pi * hourly_df['hour'] / 24)
    hourly_df['day_sin'] = np.sin(2 * np.pi * hourly_df['day_of_year'] / 365)
    hourly_df['day_cos'] = np.cos(2 * np.pi * hourly_df['day_of_year'] / 365)
    
    print("数据预处理完成。")
    return hourly_df

def create_improved_lstm_model(input_shape):
    """构建改进的LSTM模型。"""
    print("构建改进的LSTM模型...")
    
    # 使用函数式API构建更复杂的模型
    inputs = keras.layers.Input(shape=input_shape)
    
    # 第一个LSTM层
    x = keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    x = keras.layers.BatchNormalization()(x)
    
    # 第二个LSTM层
    x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    
    # 第三个LSTM层
    x = keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    
    # 全连接层
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # 输出层
    outputs = keras.layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # 使用Adam优化器，学习率调度
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse", metrics=['mae'])
    
    print(model.summary())
    return model

def create_error_trajectories_by_day(errors, horizon=24):
    """
    按天（24小时为一组）构建误差轨迹。
    """
    num_days = len(errors) // horizon
    error_trajectories = np.zeros((num_days, horizon))
    for i in range(num_days):
        error_trajectories[i, :] = errors[i * horizon : (i + 1) * horizon]
    print(f"成功构建 {num_days} 天的误差轨迹，每天 {horizon} 个点。")
    return error_trajectories

def split_and_save_datasets(trajectories: np.ndarray):
    """
    将误差轨迹矩阵划分为形状学习集D1和尺寸校准集D2，并保存到文件。

    Args:
        trajectories (np.ndarray): 完整的误差轨迹矩阵。
    """
    print("开始划分并保存数据集...")
    num_samples = trajectories.shape[0]
    
    # 计算划分点
    split_index = int(num_samples * DATASET_SPLIT_RATIO)
    
    # 进行划分
    shape_set = trajectories[:split_index]  # D1
    calibration_set = trajectories[split_index:] # D2

    # 使用Numpy的二进制格式保存
    np.save(SHAPE_SET_FILENAME, shape_set)
    np.save(CALIBRATION_SET_FILENAME, calibration_set)

    print("-" * 30)
    print("数据集已成功创建并保存！")
    print(f"形状学习集 (D1) '{SHAPE_SET_FILENAME}': {shape_set.shape}")
    print(f"尺寸校准集 (D2) '{CALIBRATION_SET_FILENAME}': {calibration_set.shape}")
    print("-" * 30)

def main():
    """主执行函数。"""
    print("--- 开始使用改进的LSTM构建温度预测误差数据集 ---")

    # 1. 准备数据
    df = load_and_preprocess_data(CSV_FILENAME)
    print(f"数据形状: {df.shape}")
    print(f"特征列: {list(df.columns)}")

    # 2. 划分训练集和验证集 (时间序列划分)
    split_fraction = 0.715
    train_split = int(split_fraction * int(df.shape[0]))
    
    df_train = df.loc[: df.index[train_split - 1]]
    df_val = df.loc[df.index[train_split] :]
    
    print(f"训练集大小: {df_train.shape}")
    print(f"验证集大小: {df_val.shape}")
    
    # 3. 数据归一化 (非常重要)
    # 使用训练集的数据来定义scaler
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(df_train)
    scaled_val_data = scaler.transform(df_val)
    
    # 保存温度列的scaler参数，以便后续反归一化
    temp_col_index = list(df.columns).index('T (degC)')
    temp_scaler_mean = scaler.mean_[temp_col_index]
    temp_scaler_scale = scaler.scale_[temp_col_index]

    # 4. 创建Keras数据集对象
    # 训练集
    train_dataset = keras.preprocessing.timeseries_dataset_from_array(
        scaled_train_data,
        targets=scaled_train_data[SEQUENCE_LENGTH:, temp_col_index],
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=True  # 打乱训练数据
    )
    # 验证集
    val_dataset = keras.preprocessing.timeseries_dataset_from_array(
        scaled_val_data,
        targets=scaled_val_data[SEQUENCE_LENGTH:, temp_col_index],
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        shuffle=False  # 验证集不打乱
    )

    # 5. 构建和训练模型
    # 从数据集中获取一个批次来确定输入形状
    for batch in train_dataset.take(1):
        inputs, _ = batch
    
    model = create_improved_lstm_model(inputs.shape[1:])
    
    print("\n开始训练改进的LSTM模型... (这可能需要较长时间)")
    
    # 定义回调函数
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # 增加耐心值
        restore_best_weights=True
    )
    
    # 学习率调度
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # 模型检查点
    checkpoint = keras.callbacks.ModelCheckpoint(
        'best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False
    )

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        verbose=1
    )
    
    # 6. 生成预测并计算误差
    print("\n在验证集上生成预测...")
    # 我们需要在整个验证集上进行预测
    val_predictions_scaled = model.predict(val_dataset, verbose=0)

    # 反归一化预测值
    val_predictions = (val_predictions_scaled * temp_scaler_scale) + temp_scaler_mean

    # 获取真实的温度值 (注意要对齐)
    real_temperatures = df_val['T (degC)'].values[SEQUENCE_LENGTH:]

    # 计算误差
    errors = (val_predictions.flatten() - real_temperatures).flatten()
    print("误差计算完成。")
    
    # 7. 构建N x T的误差轨迹矩阵（按天分组）
    error_trajectories = create_error_trajectories_by_day(errors, HORIZON_T)

    # 8. 保存数据集
    np.save(OUTPUT_FILENAME, error_trajectories)
    print(f"\n成功！基于改进LSTM的误差轨迹数据集已保存至 '{OUTPUT_FILENAME}'")
    print(f"数据集维度: {error_trajectories.shape}")
    
    # 9. 拆分并保存数据集
    split_and_save_datasets(error_trajectories)
    
    print("--- 任务完成 ---")

if __name__ == "__main__":
    main()

