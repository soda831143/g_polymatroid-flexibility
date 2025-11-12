import numpy as np
import pandas as pd

CSV_FILENAME = "data\\weather_data_filtered.csv"
HORIZON_T = 24
DATASET_SPLIT_RATIO = 0.5

SHAPE_SET_FILENAME = "shape_set_yesterday.npy"
CALIBRATION_SET_FILENAME = "calibration_set_yesterday.npy"
PREDICTED_TEMPS_FILENAME = "yesterday_predicted_temps.npy"
TEMPS_BY_DAY_FILENAME = "temps_by_day.npy"
ALL_ERRORS_FILENAME = "all_errors_yesterday.npy"  # 新增：保存全部预测误差数据

def main():
    print("加载数据...")
    df = pd.read_csv(CSV_FILENAME)
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('utc_timestamp', inplace=True)
    temp_series = df['GR_temperature'].resample('h').mean()
    temp_series = temp_series.dropna()
    temps = temp_series.values

    # 按天分组
    num_days = len(temps) // HORIZON_T
    temps = temps[:num_days * HORIZON_T]
    temps_by_day = temps.reshape((num_days, HORIZON_T))
    np.save(TEMPS_BY_DAY_FILENAME, temps_by_day)  # 保存原始温度数据

    # 昨日预测
    predicted_by_day = np.zeros_like(temps_by_day)
    predicted_by_day[1:] = temps_by_day[:-1]
    predicted_by_day[0] = np.nan  # 第一天无法预测

    # 误差
    errors_by_day = predicted_by_day - temps_by_day

    # 去掉第一天
    temps_by_day = temps_by_day[1:]
    predicted_by_day = predicted_by_day[1:]
    errors_by_day = errors_by_day[1:]
    num_days = temps_by_day.shape[0]

    # 划分 shape_set 和 calibration_set
    split_index = int(num_days * DATASET_SPLIT_RATIO)
    shape_set = errors_by_day[:split_index]
    calibration_set = errors_by_day[split_index:]

    # 保存
    # np.save(SHAPE_SET_FILENAME, shape_set)
    # np.save(CALIBRATION_SET_FILENAME, calibration_set)
    np.save(PREDICTED_TEMPS_FILENAME, predicted_by_day)
    np.save(ALL_ERRORS_FILENAME, errors_by_day)  # 保存全部预测误差数据

    print(f"shape_set: {shape_set.shape}, calibration_set: {calibration_set.shape}")
    print(f"预测温度 shape: {predicted_by_day.shape}")
    print(f"原始温度 shape: {temps_by_day.shape}")
    print(f"全部预测误差 shape: {errors_by_day.shape}")
    print("保存完毕。")

if __name__ == "__main__":
    main() 