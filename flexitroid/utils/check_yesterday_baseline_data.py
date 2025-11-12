import numpy as np
import matplotlib.pyplot as plt
import os

# 文件名
SHAPE_SET_FILENAME = 'shape_set_yesterday.npy'
CALIBRATION_SET_FILENAME = 'calibration_set_yesterday.npy'
PREDICTED_TEMPS_FILENAME = 'yesterday_predicted_temps.npy'
TEMPS_BY_DAY_FILENAME = 'temps_by_day.npy'

# 检查文件是否存在
def check_file(filename):
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return False
    return True

def print_stats(name, arr):
    print(f"--- {name} ---")
    print(f"shape: {arr.shape}")
    print(f"mean: {np.mean(arr):.4f}, std: {np.std(arr):.4f}")
    print(f"min: {np.min(arr):.4f}, max: {np.max(arr):.4f}")
    print(f"前两天数据:\n{arr[:2]}")
    print()

def main():
    # 加载并检查数据
    if check_file(SHAPE_SET_FILENAME):
        shape_set = np.load(SHAPE_SET_FILENAME)
        print_stats('shape_set', shape_set)
        plt.figure()
        plt.plot(shape_set[0], label='第1天误差')
        plt.plot(shape_set[1], label='第2天误差')
        plt.title('shape_set 前两天误差')
        plt.legend()
    if check_file(CALIBRATION_SET_FILENAME):
        calibration_set = np.load(CALIBRATION_SET_FILENAME)
        print_stats('calibration_set', calibration_set)
        plt.figure()
        plt.plot(calibration_set[0], label='第1天误差')
        plt.plot(calibration_set[1], label='第2天误差')
        plt.title('calibration_set 前两天误差')
        plt.legend()
    if check_file(PREDICTED_TEMPS_FILENAME):
        predicted_temps = np.load(PREDICTED_TEMPS_FILENAME)
        print_stats('predicted_temps', predicted_temps)
        plt.figure()
        plt.plot(predicted_temps[0], label='第1天预测')
        plt.plot(predicted_temps[1], label='第2天预测')
        plt.title('predicted_temps 前两天')
        plt.legend()
    if check_file(TEMPS_BY_DAY_FILENAME):
        temps_by_day = np.load(TEMPS_BY_DAY_FILENAME)
        print_stats('temps_by_day', temps_by_day)
        plt.figure()
        plt.plot(temps_by_day[0], label='第1天真实')
        plt.plot(temps_by_day[1], label='第2天真实')
        plt.title('temps_by_day 前两天')
        plt.legend()
    plt.show()

if __name__ == '__main__':
    main() 