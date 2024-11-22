# 定义数据集
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class GSPDataset(Dataset):
    def __init__(self, data,                     
                gsp_frequency_per_hour,
                gsp_input_time_len_in_hour,
                output_time_len_in_hour,
                predict_interval_in_hour):
        
        self.data = data
        self.gsp_frequency_per_hour = gsp_frequency_per_hour
        self.gsp_input_time_len_in_hour = gsp_input_time_len_in_hour
        self.output_time_len_in_hour = output_time_len_in_hour
        self.predict_interval_in_hour = predict_interval_in_hour

        self.gsp_seq_length = gsp_input_time_len_in_hour*gsp_frequency_per_hour
        self.gsp_forecast_horizon = (output_time_len_in_hour+1)*gsp_frequency_per_hour
        self.gsp_predict_interval = predict_interval_in_hour*gsp_frequency_per_hour

    def __len__(self):
        return (len(self.data) - self.gsp_seq_length - self.gsp_forecast_horizon)//self.gsp_predict_interval + 1

    def __getitem__(self, idx):
        if isinstance(self.data, pd.DataFrame):
            gsp_seq = self.data[idx*self.gsp_predict_interval : 
                        idx*self.gsp_predict_interval+self.gsp_seq_length].values.astype('float32')
            gsp_label = self.data[idx*self.gsp_predict_interval+self.gsp_seq_length : 
                        idx*self.gsp_predict_interval+self.gsp_seq_length+self.gsp_forecast_horizon : 
                        4].values.astype('float32')
        else:
            gsp_seq = self.data[idx*self.gsp_predict_interval : 
                        idx*self.gsp_predict_interval+self.gsp_seq_length].astype('float32')
            gsp_label = self.data[idx*self.gsp_predict_interval+self.gsp_seq_length : 
                        idx*self.gsp_predict_interval+self.gsp_seq_length+self.gsp_forecast_horizon : 
                        4].astype('float32')
        return gsp_seq, gsp_label


if __name__=='__main__':
    import pandas as pd
    gsp_frequency_per_hour = 4 # 数据每15分钟一帧
    gsp_input_time_len_in_hour = 24 # 输入的时长
    output_time_len_in_hour = 4 #要预测的时长，比如预测4小时，那就给出5个结果，【0，1，2，3，4】小时的结果
    predict_interval_in_hour = 4  # 每四个小时预测一次，每次预测都用预测启动时间点的前24小时作为输入，预测启动时间点的后4小时作为输出


    df = pd.read_csv('.//exemple_data//gfs//SQ2.csv').head(20000)
    df = df[['generation(kWh)']]
    # 替换非数字值为 NaN, 前向填充
    df = df.applymap(lambda x: np.nan if not isinstance(x, (int, float)) and pd.isna(x) == False and x != '' else x)
    df = df.ffill()
    data = np.array(df)
    dataset = GSPDataset(data, 
                        gsp_frequency_per_hour = gsp_frequency_per_hour,
                        gsp_input_time_len_in_hour = gsp_input_time_len_in_hour,
                        output_time_len_in_hour = output_time_len_in_hour,
                        predict_interval_in_hour = predict_interval_in_hour,)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for i,batch in enumerate(dataloader):
        if i==1:
            print(batch)  # batch是一个元组，包含(seq, pic)和label
            break
    print('done',batch[0].shape)

