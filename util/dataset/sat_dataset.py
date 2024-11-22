import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class SATDataset(Dataset):
    def __init__(self, data,                     
                sat_frequency_per_hour,
                sat_input_time_len_in_hour,
                shifting_hour):
        
        self.data = data
        self.sat_frequency_per_hour = sat_frequency_per_hour
        self.sat_input_time_len_in_hour = sat_input_time_len_in_hour
        self.shifting_hour = shifting_hour #时间滑移频率

        self.sat_seq_length = sat_input_time_len_in_hour*sat_frequency_per_hour # 一个数据有几个时间帧
 
    def __len__(self):
        return  int(np.ceil ((1//self.sat_frequency_per_hour)*(self.data.shape[2] - self.sat_seq_length ))//self.shifting_hour+1)

    def __getitem__(self, idx):
        #这里有一个思维陷阱：这个idx实际是gsp的idx。
        if isinstance(self.data, np.ndarray) or isinstance(self.data, torch.Tensor):
             # sat 数据是4维：【channel，time，W，H】，我们在时间上切片

            time_slice_1 = int((idx*self.shifting_hour-1)//(1/self.sat_frequency_per_hour) +1)
            time_slice_2 = int(time_slice_1 + self.sat_seq_length)
            # print(time_slice_1,time_slice_2)
            sat_seq = self.data[:, time_slice_1:time_slice_2, :, :].astype('float32') 
        else:
           raise TypeError('nwp data must be ndarray or torch array')
        return sat_seq

if __name__=='__main__':    
    # 准备数据

    sat_frequency_per_hour = 1/6 # 数据每六小时一帧
    sat_input_time_len_in_hour = 24 # 输入的时长
    shifting_hour = 4

    import numpy as np

    sat_data = np.load('.//jq_data//jiuquan_sat.npy', allow_pickle=True)

    sat_data = SATDataset(sat_data,                     
                        sat_frequency_per_hour = sat_frequency_per_hour,
                        sat_input_time_len_in_hour = sat_input_time_len_in_hour,
                        shifting_hour = shifting_hour)
    dataloader = DataLoader(sat_data, batch_size=1, shuffle=False)
    for i,batch in enumerate(dataloader):
        print(batch.shape)
        break

        


