import torch
import torch.nn as nn
import numpy as np
from  model.basemodel import  BaseModel
from  model.encoder.concat_encoder import ConcatModel
from  model.encoder.gsp_encoder import Conv1d_GSP
from  model.encoder.nwp_encoder import Conv3d_NWP
from  model.encoder.sat_encoder import Conv3d_SAT
from  model.encoder.ec_encoder import Conv3d_EC

class MultiModel(BaseModel):
    def __init__(self, config ):
        super(MultiModel, self).__init__()
        self.config = config

        self.conv1d_gsp = Conv1d_GSP(in_channels=config.gsp_in_channels,  
                                    kernel_sizes=config.gsp_kernel_sizes,
                                    paddings=config.gsp_paddings, 
                                    channel_num_list = config.gsp_channel_num_list, 
                                    tf_input_dim = config.gsp_tf_input_dim,
                                    nhead=config.gsp_nhead, 
                                    nhid=config.gsp_nhid, 
                                    nlayers=config.gsp_nlayers, 
                                    dropout=0.3)
        if config.include_nwp:
            self.conv3d_nwp_list = []
            for i in range(config.nwp_source_num):
                conv3d_nwp = Conv3d_NWP(in_channels=config.nwp_in_channels[i],  
                                    kernel_sizes=config.nwp_kernel_sizes[i],
                                    paddings=config.nwp_paddings[i], 
                                    channel_num_list = config.nwp_channel_num_list[i], 
                                    tf_input_dim = config.nwp_tf_input_dim[i],
                                    nhead=config.nwp_nhead[i], 
                                    nhid=config.nwp_nhid[i], 
                                    nlayers=config.nwp_nlayers[i], 
                                    dropout=0.3)
                self.conv3d_nwp_list.append(conv3d_nwp)
                
        self.concatmodel = ConcatModel(input_feature=config.cm_input_feature, 
                                    input_len=config.cm_input_len, 
                                    output_feature=config.cm_output_feature, 
                                    ouput_len=config.ouput_len)
        self.train_losses = []
            
 


    def forward(self, multi_data):
        if self.config.include_nwp:
            if self.config.include_sat:
                if self.config.include_ec:
                    gsp_matrix, nwp_matrixs, sat_matrixs, ec_matrix = multi_data
                else:
                    gsp_matrix, nwp_matrixs, sat_matrixs = multi_data
            else:
                if self.config.include_ec:
                    gsp_matrix, nwp_matrixs, ec_matrix = multi_data
                else:
                    gsp_matrix, nwp_matrixs = multi_data
        else:
            if self.config.include_sat:
                if self.config.include_ec:
                    gsp_matrix, sat_matrixs, ec_matrix = multi_data
                else:
                    gsp_matrix, sat_matrixs = multi_data
            else:
                if self.config.include_ec:
                    gsp_matrix, ec_matrix = multi_data
                else:
                    gsp_matrix = multi_data

        gsp_output = self.conv1d_gsp(gsp_matrix)
        gsp_output_list = [gsp_output]

        if self.config.include_nwp:
            nwp_output_list=[]
            for i, conv3d_nwp in enumerate(self.conv3d_nwp_list):
                nwp_output = conv3d_nwp(nwp_matrixs[i])
                nwp_output_list.append(nwp_output)

        if self.config.include_sat:
            sat_output_list=[]
            for i, conv3d_sat in enumerate(self.conv3d_sat_list):
                sat_output = conv3d_sat(sat_matrixs[i])
                sat_output_list.append(sat_output)

        if self.config.include_ec:
            ec_output = self.conv3d_ec(ec_matrix[i])
            ec_output_list = [ec_output]

        if self.config.include_nwp:
            if self.config.include_sat:
                if self.config.include_ec:
                    concated_ = self.concatmodel(gsp_output_list,
                                                 nwp_output_list,
                                                 sat_output_list,
                                                 ec_output_list  )
                else:
                    concated_ = self.concatmodel(gsp_output_list,
                                                 nwp_output_list,
                                                 sat_output_list, )
            else:
                if self.config.include_ec:
                    concated_ = self.concatmodel(gsp_output_list,
                                                 nwp_output_list,
                                                 ec_output_list, )
                else:
                    concated_ = self.concatmodel(gsp_output_list,
                                                 nwp_output_list, )
        else:
            if self.config.include_sat:
                if self.config.include_ec:
                    concated_ = self.concatmodel(gsp_output_list,
                                                 sat_output_list,
                                                 ec_output_list  )
                else:
                    concated_ = self.concatmodel(gsp_output_list,
                                                 sat_output_list)
            else:
                if self.config.include_ec:
                    concated_ = self.concatmodel(gsp_output_list,
                                                 ec_output_list  )
                else:
                    concated_ = self.concatmodel(gsp_output_list)
        return concated_


if __name__=='__main__':
    from util.test_config import Config
    config = Config()
    config.gsp_in_channels = 1
    config.gsp_channel_num_list = [4,8,16,32,32,32]
    config.gsp_kernel_sizes = [3]*len(config.channel_num_list)
    config.gsp_paddings = [1]*len(config.channel_num_list)
    config.gsp_tf_input_dim = 72
    config.gsp_nhead = 36
    config.gsp_nhid = 10
    config.gsp_nlayers = 6

    

    # batch_num = 16
    # input_size = 96
    # in_channels = 1
    # gsp_matrix = np.random.rand(batch_num, input_size, in_channels) 

    # multi_data = (gsp_matrix)

    # output = multimodel(multi_data)
    # print('output shape',output.shape)


    import pandas as pd    
    from torch.utils.data import Dataset, DataLoader
    from util.dataset.gsp_dataset_single import GSPDataset
    # 准备数据
    gsp_frequency_per_hour = 4 # 数据每15分钟一帧
    gsp_input_time_len_in_hour = 24 # 输入的时长
    output_time_len_in_hour = 4 #要预测的时长，比如预测4小时，那就给出5个结果，【0，1，2，3，4】小时的结果
    predict_interval_in_hour = 4  # 每四个小时预测一次，每次预测都用预测启动时间点的前24小时作为输入，预测启动时间点的后4小时作为输出
    # data = {
    #     'generation(kWh)': [i*0.25 for i in range(1000)],  # 用你的实际数据替换...
    #     'power(W)': [i*0.25*10 for i in range(1000)]  # 用你的实际数据替换...
    # }
    # df = pd.DataFrame(data)
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

    # # 使用for循环迭代数据加载器
    # for i,batch in enumerate(dataloader):
    #     if i==1:
    #         print(batch)  # batch是一个元组，包含(seq, pic)和label
    #         break
    # # 实例化模型并训练
    model = MultiModel(config)
    import pytorch_lightning as pl
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, dataloader)



 




    
