import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def minmax(y,t):
    scle = MinMaxScaler()
    scle.fit(t)
    t = scle.transform(t)
    y = scle.transform(y)
    return y,t

def train_test_pic(train_predictions ,train_truth ,test_predictions ,test_truth ):
    plt.plot(train_predictions, label='prediction')
    plt.plot(train_truth, label='truth')
    plt.xlabel('time')
    plt.title('training pic')
    plt.legend()
    plt.show()
    
    plt.plot(test_predictions, label='prediction')
    plt.plot(test_truth, label='truth')
    plt.xlabel('time')
    plt.title('testing pic')
    plt.legend()
    plt.show()

    y1,t1 = minmax(train_predictions ,train_truth)
    y2,t2 = minmax(test_predictions ,test_truth)

    mae1 = calculate_mae(t1, y1)
    mae2 = calculate_mae(t2, y2)

    mae = np.abs(y1 - t1)

    plt.figure(figsize=(10, 6))
    plt.plot(mae, label='MAE between pred and true', color='r', linewidth=2)
    plt.title(f'MAE of traing')
    plt.xlabel('time')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()

    mae = np.abs(y2 - t2)

    plt.figure(figsize=(10, 6))
    plt.plot(mae, label='MAE between pred and true', color='r', linewidth=2)
    plt.title(f'MAE of testing')
    plt.xlabel('time')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f'训练数据MAE为：{mae1*100}%,测试数据MAE为：{mae2*100}%')

def compare_pic(test_predictions ,test_truth ):
    length = test_predictions.shape[0]
    com_data = np.load('./jq_data/com_data.npy')
    com_test = com_data[-length*4:]
    
    time_1h = np.arange(0, length)  
    time_15min = np.arange(0, length, 0.25)
    
    interpolator = interp1d(time_1h, test_predictions.flatten(), kind='linear', fill_value='extrapolate')
    yy = interpolator(time_15min)
    
    interpolator = interp1d(time_1h, test_truth.flatten(), kind='linear', fill_value='extrapolate')
    tt = interpolator(time_15min)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_15min, yy, label="Prediction", color='b')
    plt.plot(time_15min, tt, label="Truth", color='y')
    plt.plot(time_15min, com_test, label="Ultra-short term ", color='r', linestyle='--')

    plt.xlabel("Time (hours)")
    plt.ylabel("Value")
    plt.title("Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    yy = yy.reshape(length*4,1)
    tt = tt.reshape(length*4,1)

    y1,t = minmax(yy ,tt )
    y2,t = minmax(com_test ,tt )

    p_mae = np.abs(y1 - t)
    u_mae = np.abs(y2 - t)

    plt.figure(figsize=(10, 6))
    plt.plot(p_mae, label='MAE of msnet', color='r', linewidth=2)
    plt.plot(u_mae, label='MAE of now', color='b', linewidth=2)
    plt.title('MAE Between Two Curves')
    plt.xlabel('time')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()

    mae1 = calculate_mae(t, y1)
    mae2 = calculate_mae(t, y2)
    
    print(f'现有模型MAE为：{mae2*100}%,我方模型MAE为：{mae1*100}%')
    