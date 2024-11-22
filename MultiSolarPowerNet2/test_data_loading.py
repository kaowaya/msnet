from  util.dataset.multi_dataset import PV_multi_Dataset
from  util.test_config import Config
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

if __name__=='__main__':

    config = Config()
    print(config.include_nwp)
    config.include_nwp = True
    print(config.include_nwp)
    # config.gsp_frequency_per_hour=4
    # config.gsp_input_time_len_in_hour=24
    # config.output_time_len_in_hour=4
    # config.predict_interval_in_hour=4
    # config.nwp_frequency_per_hour=1/6
    # config.nwp_input_time_len_in_hour=24
    # config.shifting_hour=4


    df = pd.read_csv('.//exemple_data//gfs//SQ2.csv').head(20000)
    df = df[['generation(kWh)']]
    # 替换非数字值为 NaN, 前向填充
    df = df.applymap(lambda x: np.nan if not isinstance(x, (int, float)) and pd.isna(x) == False and x != '' else x)
    df = df.ffill()
    gsp_data = np.array(df)

    nwp_data = np.load('.//exemple_data//nwp//nwp_data_202207.npy', allow_pickle=True)
    sat_data = np.load('.//jq_data//jiuquan_sat.npy', allow_pickle=True)

    data = gsp_data,[nwp_data],[sat_data]
    dataset = PV_multi_Dataset(config, data)  
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)    
    for batch in dataloader:
        print(batch[0][0].shape) 
        print(batch[0][1][0].shape) # nwp
        print(batch[0][2][0].shape) # sat
        break