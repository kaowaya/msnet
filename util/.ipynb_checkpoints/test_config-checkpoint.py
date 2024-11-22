class Config():
    def __init__(self):
        self.config_path = "./configs/"
        self.config_name = "config.yaml"
        self.read_config()
        self.parse_config()

    def read_config(self):
        import yaml

        # 打开并读取 YAML 文件# self.config_path + self.config_name
        with open('.\config\config.yaml', "r") as file:
            data = yaml.safe_load(file)

        if isinstance(data['basic'],dict):
            self.dict_basic = data['basic']
        elif isinstance(data['basic'],list):
            self.dict_basic = {key: value for d in data['basic'] for key, value in d.items()}

        if isinstance(data['datamodule'],dict):
            self.dict_datamodule = data['datamodule']
        elif isinstance(data['datamodule'],list):
            self.dict_datamodule = {key: value for d in data['datamodule'] for key, value in d.items()}

        if isinstance(data['dataset'],dict):
            self.dict_dataset = data['dataset']
        elif isinstance(data['dataset'],list):
            self.dict_dataset = {key: value for d in data['dataset'] for key, value in d.items()}

        if isinstance(data['training'],dict):
            self.dict_training = data['training']
        elif isinstance(data['training'],list):
            self.dict_training = {key: value for d in data['training'] for key, value in d.items()}

 
    def parse_config(self):
        self.include_nwp = self.dict_basic['include_nwp']
        self.include_sat = self.dict_basic['include_sat']
        self.include_ec =  self.dict_basic['include_ec']

        self.gsp_frequency_per_hour=self.dict_dataset['gsp_frequency_per_hour']
        self.gsp_input_time_len_in_hour=self.dict_dataset['gsp_input_time_len_in_hour']
        self.output_time_len_in_hour=self.dict_dataset['output_time_len_in_hour']
        self.predict_interval_in_hour=self.dict_dataset['predict_interval_in_hour']

        self.shifting_hour=self.dict_dataset['shifting_hour']

        if self.include_nwp:
            self.nwp_source_num = self.dict_basic['nwp_source_num']
            self.nwp_frequency_per_hour=eval(self.dict_dataset['nwp_frequency_per_hour'])
            self.nwp_input_time_len_in_hour=self.dict_dataset['nwp_input_time_len_in_hour']
            
        if self.include_sat:
            self.sat_frequency_per_hour=eval(self.dict_dataset['sat_frequency_per_hour'])
            self.sat_input_time_len_in_hour=self.dict_dataset['sat_input_time_len_in_hour']

        if self.include_ec:
            self.ec_frequency_per_hour=eval(self.dict_dataset['ec_frequency_per_hour'])
            self.ec_input_time_len_in_hour=self.dict_dataset['ec_input_time_len_in_hour']

        self.gsp_in_channels = self.dict_training['gsp_in_channels']
        self.gsp_kernel_sizes = self.dict_training['gsp_kernel_sizes']
        self.gsp_paddings = self.dict_training['gsp_paddings']
        self.gsp_channel_num_list = self.dict_training['gsp_channel_num_list']
        self.gsp_tf_input_dim = self.dict_training['gsp_tf_input_dim']
        self.gsp_nhead = self.dict_training['gsp_nhead']
        self.gsp_nhid = self.dict_training['gsp_nhid']
        self.gsp_nlayers = self.dict_training['gsp_nlayers']

        self.cm_input_feature = self.dict_training['cm_input_feature']
        self.cm_input_len = self.dict_training['cm_input_len']
        self.cm_output_feature  = self.dict_training['cm_output_feature']
        self.ouput_len = self.dict_training['ouput_len']

        if self.include_nwp:
            self.nwp_in_channels = self.dict_training['nwp_in_channels']
            self.nwp_kernel_sizes = self.dict_training['nwp_kernel_sizes']
            self.nwp_paddings = self.dict_training['nwp_paddings']
            self.nwp_channel_num_list = self.dict_training['nwp_channel_num_list']
            self.nwp_tf_input_dim = self.dict_training['nwp_tf_input_dim']
            self.nwp_nhead = self.dict_training['nwp_nhead']
            self.nwp_nhid = self.dict_training['nwp_nhid']
            self.nwp_nlayers = self.dict_training['nwp_nlayers']

        if self.include_sat:
            self.sat_in_channels = self.dict_training['sat_in_channels']
            self.sat_kernel_sizes = self.dict_training['sat_kernel_sizes']
            self.sat_paddings = self.dict_training['sat_paddings']
            self.sat_channel_num_list = self.dict_training['sat_channel_num_list']
            self.sat_tf_input_dim = self.dict_training['sat_tf_input_dim']
            self.sat_nhead = self.dict_training['sat_nhead']
            self.sat_nhid = self.dict_training['sat_nhid']
            self.sat_nlayers = self.dict_training['sat_nlayers']
        
        if self.include_ec:
            self.ec_in_channels = self.dict_training['ec_in_channels']
            self.ec_kernel_sizes = self.dict_training['ec_kernel_sizes']
            self.ec_paddings = self.dict_training['ec_paddings']
            self.ec_channel_num_list = self.dict_training['ec_channel_num_list']
            self.ec_tf_input_dim = self.dict_training['ec_tf_input_dim']
            self.ec_nhead = self.dict_training['ec_nhead']
            self.ec_nhid = self.dict_training['ec_nhid']
            self.ec_nlayers = self.dict_training['ec_nlayers']





if __name__=='__main__':
    c = Config()
    c.cm_input_feature