import numpy as np
from torch.utils.data import Dataset, DataLoader
from util.dataset.gsp_dataset_single import GSPDataset
from util.dataset.nwp_dataset import NWPDataset
from util.dataset.sat_dataset import SATDataset

# 定义数据集
class PV_multi_Dataset(Dataset):
    def __init__(self,config, data):
        super(PV_multi_Dataset, self).__init__()
        self.config = config
        if config.include_nwp:
            print('include_nwp')
            gsp_data, nwp_data_list,sat_data_list = data
        else:
            print('not include_nwp')
            gsp_data = data
        
        self.gsp_dataset = GSPDataset(gsp_data, 
                            gsp_frequency_per_hour = config.gsp_frequency_per_hour,
                            gsp_input_time_len_in_hour = config.gsp_input_time_len_in_hour,
                            output_time_len_in_hour = config.output_time_len_in_hour,
                            predict_interval_in_hour = config.predict_interval_in_hour,)
        
        if config.include_nwp:
            self.nwp_dataset_list = []
            for nwp_data in nwp_data_list:
                nwp_dataset = NWPDataset(nwp_data,                     
                            nwp_frequency_per_hour = config.nwp_frequency_per_hour,
                            nwp_input_time_len_in_hour = config.nwp_input_time_len_in_hour,
                            shifting_hour = config.shifting_hour)
                self.nwp_dataset_list .append(nwp_dataset)
        if config.include_sat:
            self.sat_dataset_list = []
            for sat_data in sat_data_list:
                sat_dataset = SATDataset(sat_data,                     
                            sat_frequency_per_hour = config.sat_frequency_per_hour,
                            sat_input_time_len_in_hour = config.sat_input_time_len_in_hour,
                            shifting_hour = config.shifting_hour)
                self.sat_dataset_list .append(sat_dataset)

    def __len__(self):
        return self.gsp_dataset.__len__()

    def __getitem__(self, idx):

        x_list = []
        gsp_seq, label = self.gsp_dataset.__getitem__(idx)
        x_list.append(gsp_seq)

        if self.config.include_nwp:
            nwp_seq_list = []
            for nwp_dataset in self.nwp_dataset_list:
                nwp_seq = nwp_dataset.__getitem__(idx)
                nwp_seq_list.append(nwp_seq)
            x_list.append(nwp_seq_list)

        if self.config.include_sat:
            sat_seq_list = []
            for sat_dataset in self.sat_dataset_list:
                sat_seq = sat_dataset.__getitem__(idx)
                sat_seq_list.append(sat_seq)
            x_list.append(sat_seq_list)
        # for sat_dataset in self.sat_dataset_list:
        #     sat_seq = sat_dataset.__getitem__(idx)
        #     x_list.append(sat_seq)



        return x_list, label





if __name__=='__main__':
    from gsp_dataset_single import GSPDataset
    from nwp_dataset import NWPDataset
    from ..test_config import Config
    
    dataset = PV_multi_Dataset()  
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)    
    for batch in dataloader:
        print(batch[0][0].shape) 
        print(batch[0][1][0].shape) # batch是一个元组，包含(seq, pic)和label
        break