import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class NWPDataset(Dataset):
    def __init__(self, data,                     
                nwp_frequency_per_hour,
                nwp_input_time_len_in_hour,
                shifting_hour):
        
        self.data = data
        self.nwp_frequency_per_hour = nwp_frequency_per_hour
        self.nwp_input_time_len_in_hour = nwp_input_time_len_in_hour
        self.shifting_hour = shifting_hour #时间滑移频率

        self.nwp_seq_length = nwp_input_time_len_in_hour*nwp_frequency_per_hour # 一个数据有几个时间帧
 
    def __len__(self):
        return  int(np.ceil ((1//self.nwp_frequency_per_hour)*(self.data.shape[2] - self.nwp_seq_length ))//self.shifting_hour+1)

    def __getitem__(self, idx):
        #这里有一个思维陷阱：这个idx实际是gsp的idx。
        if isinstance(self.data, np.ndarray) or isinstance(self.data, torch.Tensor):
             # nwp 数据是五维：【channel，step，time，W，H】，我们在时间上切片

            time_slice_1 = int((idx*self.shifting_hour-1)//(1/self.nwp_frequency_per_hour) +1)
            time_slice_2 = int(time_slice_1 + self.nwp_seq_length)
            # print(time_slice_1,time_slice_2)
            nwp_seq = self.data[:, :, time_slice_1:time_slice_2, :, :].astype('float32') 
        else:
           raise TypeError('nwp data must be ndarray or torch array')
        return nwp_seq

if __name__=='__main__':    
    # 准备数据

    nwp_frequency_per_hour = 1/6 # 数据每六小时一帧
    nwp_input_time_len_in_hour = 24 # 输入的时长
    shifting_hour = 4

    import numpy as np

    data = np.load('.//exemple_data//nwp//nwp_data_202207.npy', allow_pickle=True)

    nwp_data = NWPDataset(data,                     
                        nwp_frequency_per_hour = nwp_frequency_per_hour,
                        nwp_input_time_len_in_hour = nwp_input_time_len_in_hour,
                        shifting_hour = shifting_hour)
    dataloader = DataLoader(nwp_data, batch_size=1, shuffle=False)
    for i,batch in enumerate(dataloader):
        print(batch.shape)
        break

        


