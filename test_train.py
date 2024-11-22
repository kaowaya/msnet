import pandas as pd    
import numpy as np
from torch.utils.data import Dataset, DataLoader
from util.dataset.gsp_dataset_single import GSPDataset
from util.dataset.multi_dataset import PV_multi_Dataset
from util.test_config import Config
from model.multimodel import MultiModel

config = Config()

# 准备数据
gsp_frequency_per_hour = 4 # 数据每15分钟一帧
gsp_input_time_len_in_hour = 24 # 输入的时长
output_time_len_in_hour = 4 #要预测的时长，比如预测4小时，那就给出5个结果，【0，1，2，3，4】小时的结果
predict_interval_in_hour = 4  # 每四个小时预测一次，每次预测都用预测启动时间点的前24小时作为输入，预测启动时间点的后4小时作为输出
df = pd.read_csv('.//exemple_data//gfs//SQ2.csv').head(20000)
df = df[['generation(kWh)']]
# 替换非数字值为 NaN, 前向填充
df = df.applymap(lambda x: np.nan if not isinstance(x, (int, float)) and pd.isna(x) == False and x != '' else x)
df = df.ffill()

dataset = GSPDataset(df, 
                    gsp_frequency_per_hour = gsp_frequency_per_hour,
                    gsp_input_time_len_in_hour = gsp_input_time_len_in_hour,
                    output_time_len_in_hour = output_time_len_in_hour,
                    predict_interval_in_hour = predict_interval_in_hour,)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

multimodel = MultiModel(config)
import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=10)
trainer.fit(multimodel, dataloader)
