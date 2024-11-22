import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatModel(nn.Module):
    def __init__(self, input_feature, input_len, output_feature, ouput_len ):
        super(ConcatModel, self).__init__()
        assert input_feature == output_feature, "input_feature must be equal to output_feature"
        assert input_len >= ouput_len, "input_len must be greater than ouput_len"

        self.output_feature = output_feature
        self.input_len = input_len #len是总长度
        self.ouput_len = ouput_len

        # 定义Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_len, 
                                                     nhead=input_len, 
                                                     dim_feedforward=32, 
                                                     dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.fc1 = nn.Linear(input_len, ouput_len)

    def forward(self, *input_encodings):
        encodings = list(input_encodings) #这个list的每个元素都是一个list,子list的每个元素是一个encoding
                                          #（比如有一个gsp，两个nwp，三个sat，那就是 [[gsp], [nwp1,nwp2],  [sat1,sat2,sat3] ]   input_len=5*6
        assert len(encodings) > 0, "there must be atleast one encoding"
        if self.input_len==self.ouput_len:
            # print('ConcatModel原样输出')
            x_concat = encodings[0][0] # [16, 1, 5]
        else:
            batch_size = len(encodings[0][0])
            x_concat = torch.empty(batch_size, self.output_feature, 0)
            for encodings_this_catogory in encodings:
                for encoding in encodings_this_catogory:
                    x_concat = torch.cat((x_concat, encoding), dim=2)
            print('x_concat.shape',x_concat.shape)

            x_concat = self.transformer_encoder(x_concat)
            x_concat = F.relu(self.fc1(x_concat))
            res_connect=False
            if res_connect:
                # 将第一个 encoding 加到 transformer 输出中 (resnet 风格的短路连接)
                # 假设第一个 encoding 的形状是 [batch_size, features, seq_len]
                first_encoding = encodings[0][0]  # 取得第一个 encoding
                first_encoding = first_encoding.permute(2, 0, 1)  # 转换为 [seq_len, batch_size, features]
                
                # 确保第一个 encoding 的形状和 Transformer 输出的形状一致
                assert first_encoding.shape == x_concat.shape, "The shapes of the shortcut and transformer output must match"
                
                # 进行短路连接
                x_concat = x_concat + first_encoding
                
                # 通过 ReLU 激活函数进行处理
                x_concat = F.relu(x_concat)
        return x_concat
        


        
if __name__=='__main__':
    
    gsp1 = torch.rand(16,1,5)
    nwp1 = torch.rand(16,1,5)
    nwp2 = torch.rand(16,1,5)
    sat1 = torch.rand(16,1,5)

    gsp = [gsp1]
    nwp = [nwp1,nwp2]
    sat = [sat1]

    input_len=5
    
    cm = ConcatModel(input_feature=1, 
                    input_len=input_len, 
                    output_feature=1, 
                    ouput_len=5)

    output = cm(gsp,nwp,sat)
    print(output.shape)





# class ConcatModel_depre(nn.Module):
#     def __init__(self, input_feature, input_time, output_feature, ouput_time ):
#         super(ConcatModel_depre, self).__init__()
#         self.output_feature = output_feature
#         self.ouput_time = ouput_time
        
#         # 定义Transformer Encoder Layer
#         encoder_layers = nn.TransformerEncoderLayer(d_model=input_feature, nhead=8, dim_feedforward=256, dropout=0.1)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        
#         # 定义全连接层
#         self.fc1 = nn.Linear(input_feature, 64)
#         self.sigmoid = nn.Sigmoid()
#         self.fc2 = nn.Linear(64, self.output_feature * self.ouput_time)

#     def forward(self, x):
#         # 调整x的形状以匹配Transformer的输入
#         x_concat = torch.empty(0, 2, dtype=torch.float32)
#         for xi in x:
#             xi = xi.permute(2, 0, 1)  # 将形状从[1, 192, 10]变为[10, 1, 192]
#             x_concat = torch.cat((x_concat, xi), dim=1)

   
        
#         # 通过Transformer Encoder
#         x_concat = self.transformer_encoder(x_concat)
#         # 调整x的形状以匹配全连接层的输入
#         x_concat = x_concat.permute(1, 2, 0)  # 将形状从[10, 1, 192]变为[1, 192, 10]
        
#         # 通过第一个全连接层
#         x_concat = x_concat.mean(dim=2)  # 将时间维度从10减少到1
#         x_concat = self.fc1(x_concat)
#         x_concat = self.sigmoid(x_concat)
#         # 通过第二个全连接层
#         x_concat = self.fc2(x_concat)
        
#         # 调整x的形状以匹配最终的输出形状

#         x_concat = x_concat.view(1, self.output_feature, self.ouput_time)
#         return x_concat