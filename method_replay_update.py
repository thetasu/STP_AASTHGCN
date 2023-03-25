import json
import os
import numpy as np
import argparse
import configparser
import pandas as pd
import torch
import pickle
import time
from sklearn.preprocessing import StandardScaler
DEVICE = torch.device('cuda:0')
'''
# step0 添加一个全局标准化函数，后面也用这个std_method 进行数据还原
# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./configurations/ISFD21_aastgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

# adj_filename = data_config['adj_filename']
# graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
# points_per_hour = int(data_config['points_per_hour']) # Abandoned
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
# num_of_weeks = int(training_config['num_of_weeks']) # Abandoned
# num_of_days = int(training_config['num_of_days']) # Abandoned
# num_of_hours = int(training_config['num_of_hours'])  # Abandoned
'''
# get file name list in file_dir
def file_name(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file = os.path.splitext(file)[0]
            file_list.append(file)
    file_list.sort()
    return file_list

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'SSFD21':
        # SSFD-DATA-110 Nodes
        ###########SSFD Sector##############
        # Sector1
        # Code_list = ['APD','BBL','BHP','CTA-PB','ECL','LIN','RIO','SCCO','SHW','VALE'] # 10
        # Sector2
        # Code_list = ['ATVI','BIDU','CMCSA','DIS','GOOG','NFLX','NTES','T','TMUS','VZ'] # 10
        # Sector3
        # Code_list = ['AMZN','BKNG','HD','LOW','MCD','MELI','NKE','SBUX','TM','TSLA'] # 10
        # Sector4
        # Code_list = ['BUD','COST','DEO','EL','KO','PEP','PG','PM','UL','WMT'] # 10
        # Sector5
        # Code_list = ['BP','COP','CVX','ENB','EQNR','PTR','RDS-B','SNP','TOT','XOM'] # 10
        # Sector6
        # Code_list = ['BAC','BML-PG','BML-PL','BRK-B','JPM','LFC','MA','MS','V','WFC'] # 10
        # Sector7
        # Code_list = ['ABT','JNJ','LLY','MDT','MRK','NVO','NVS','PFE','TMO','UNH'] # 10
        # Sector8
        # Code_list = ['BA','CAT','DE','GE','HON','LMT','MMM','RTX','UNP','UPS'] # 10
        # Sector9
        # Code_list = ['AMT','CSGP','DLR','EQIX','PLD','SBAC','SPG','SPG-PJ','WELL','WY'] # 10
        # Sector10
        # Code_list = ['AAPL','ADBE','ASML','AVGO','CRM','CSCO','INTC','MSFT','NVDA','TSM'] # 10
        # Sector11
        # Code_list = ['AEP','D','ES','EXC','NEE','NGG','PEG','SO','SRE','XEL'] # 10
        # # Code_list = file_name('../data/SHY21-Arima-back')

        ##############################
        Code_list = file_name('./data/SSFD21-Arima')

        ###############################
        data_path = os.path.join('./data/SSFD21/SSFD-V1_11.csv')
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        for code in Code_list:
            # print(code)
            df_code = df[(df['Code'] == code)]
            df_code_adj_close = df_code['Adj Close'].values.tolist()
            df_group_data[code] = df_code_adj_close
            df_code_label = df_code['Label'].values.tolist()
            df_label_data[code] = df_code_label
        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)
        data = np.transpose([df_group_data.values])
    elif dataset == 'ISFD21':
        # SSFD-DATA-105 Nodes
        ##########
        # Sector1
        # Code_list = ['AVD','CF','CTA-PA','CTA-PB','FMC','ICL','IPI','MOS','SMG'] # 9
        # Sector2
        Code_list = ['AMOV','AMX','BCE','CHT','ORAN','T','TMUS','TU','VOD','VZ'] # 10
        # Sector3
        # Code_list = ['ALV','BWA','DAN','DORM','GNTX','GT','LEA','LKQ','MGA','VC'] # 10
        # Sector4
        # Code_list = ['ADM','ALCO','BG','CALM','CHSCP','FDP','IBA','LMNR','TSN'] # 9
        # Sector5
        # Code_list = ['CEO','CLR','CNQ','COP','DVN','EOG','HES','MRO','OXY','PXD'] # 10
        # Sector6
        # Code_list = ['AMP','BAM','BEN','BK','BLK','BX','KKR','NTRS','STT','TROW'] # 10
        # Sector7
        # Code_list = ['CERN','CPSI','HMSY','HSTM','MDRX','NXGN','OMCL'] # 7
        # Sector8
        # Code_list = ['AIT','DXPE','FAST','GWW','LAWS','MSM','PKOH','SYX','WCC','WSO'] # 10
        # Sector9
        # Code_list = ['CBRE','CIGI','CSGP','CSR','FRPH','IRCP','JLL','KW','NTP','TCI'] # 10
        # Sector10
        # Code_list = ['ADSK','ANSS','CDNS','CRM','CTXS','INTU','PTC','SAP','SSNC','TYL'] # 10
        # Sector11
        # Code_list = ['AEP','DTE','DUK','ED','ES','NEE','PCG','SO','WFC','XEL'] #10

        ##########
        # Code_list = ['CDNS','CTXS','PTC','SAP','SSNC']
        # Code_list = ['ADSK','ANSS','CRM','INTU','TYL']
        #########################################
        # Code_list = file_name('./data/ISFD21-Arima')
        data_path = os.path.join('./data/ISFD21/ISFD-V1_11.csv')
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        for code in Code_list:
            df_code = df[(df['Code'] == code)]
            df_code_adj_close = df_code['Adj Close'].values.tolist()
            df_group_data[code] = df_code_adj_close
            df_code_label = df_code['Label'].values.tolist()
            df_label_data[code] = df_code_label

        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)

        data = np.transpose([df_group_data.values])
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    # Normalization using Z-score method
    X = data
    print('X.shape:{}'.format(X.shape))
    # means = np.mean(X, axis=(0, 2))
    means = np.mean(X, axis=(0, 1))

    X = X - means.reshape(1, -1, 1)
    # stds = np.std(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 1))
    X = X / stds.reshape(1, -1, 1)

    means = means.reshape(1,1,1,1)
    stds = stds.reshape(1,1,1,1)
    print('means:{}'.format(means))
    print('stds:{}'.format(stds))
    return X,means,stds,data


def load_st_dataset_standard(dataset):
    #output B, N, D
    if dataset == 'SSFD21':
        # SSFD-DATA-110 Nodes
        ###########SSFD Sector##############
        # Sector1
        # Code_list = ['APD','BBL','BHP','CTA-PB','ECL','LIN','RIO','SCCO','SHW','VALE'] # 10
        # Sector2
        # Code_list = ['ATVI','BIDU','CMCSA','DIS','GOOG','NFLX','NTES','T','TMUS','VZ'] # 10
        # Sector3
        # Code_list = ['AMZN','BKNG','HD','LOW','MCD','MELI','NKE','SBUX','TM','TSLA'] # 10
        # Sector4
        # Code_list = ['BUD','COST','DEO','EL','KO','PEP','PG','PM','UL','WMT'] # 10
        # Sector5
        # Code_list = ['BP','COP','CVX','ENB','EQNR','PTR','RDS-B','SNP','TOT','XOM'] # 10
        # Sector6
        # Code_list = ['BAC','BML-PG','BML-PL','BRK-B','JPM','LFC','MA','MS','V','WFC'] # 10
        # Sector7
        # Code_list = ['ABT','JNJ','LLY','MDT','MRK','NVO','NVS','PFE','TMO','UNH'] # 10
        # Sector8
        # Code_list = ['BA','CAT','DE','GE','HON','LMT','MMM','RTX','UNP','UPS'] # 10
        # Sector9
        # Code_list = ['AMT','CSGP','DLR','EQIX','PLD','SBAC','SPG','SPG-PJ','WELL','WY'] # 10
        # Sector10
        # Code_list = ['AAPL','ADBE','ASML','AVGO','CRM','CSCO','INTC','MSFT','NVDA','TSM'] # 10
        # Sector11
        # Code_list = ['AEP','D','ES','EXC','NEE','NGG','PEG','SO','SRE','XEL'] # 10
        # # Code_list = file_name('../data/SHY21-Arima-back')

        ##############################
        Code_list = file_name('./data/SSFD21-Arima')

        ###############################
        data_path = os.path.join('./data/SSFD21/SSFD-V1_11.csv')
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        for code in Code_list:
            # print(code)
            df_code = df[(df['Code'] == code)]
            df_code_adj_close = df_code['Adj Close'].values.tolist()
            df_group_data[code] = df_code_adj_close
            df_code_label = df_code['Label'].values.tolist()
            df_label_data[code] = df_code_label
        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)
        data = np.transpose([df_group_data.values])
    elif dataset == 'ISFD21':
        # SSFD-DATA-105 Nodes
        ##########
        # Sector1
        # Code_list = ['AVD','CF','CTA-PA','CTA-PB','FMC','ICL','IPI','MOS','SMG'] # 9
        # Sector2
        # Code_list = ['AMOV','AMX','BCE','CHT','ORAN','T','TMUS','TU','VOD','VZ'] # 10
        # Sector3
        # Code_list = ['ALV','BWA','DAN','DORM','GNTX','GT','LEA','LKQ','MGA','VC'] # 10
        # Sector4
        # Code_list = ['ADM','ALCO','BG','CALM','CHSCP','FDP','IBA','LMNR','TSN'] # 9
        # Sector5
        # Code_list = ['CEO','CLR','CNQ','COP','DVN','EOG','HES','MRO','OXY','PXD'] # 10
        # Sector6
        # Code_list = ['AMP','BAM','BEN','BK','BLK','BX','KKR','NTRS','STT','TROW'] # 10
        # Sector7
        # Code_list = ['CERN','CPSI','HMSY','HSTM','MDRX','NXGN','OMCL'] # 7
        # Sector8
        # Code_list = ['AIT','DXPE','FAST','GWW','LAWS','MSM','PKOH','SYX','WCC','WSO'] # 10
        # Sector9
        # Code_list = ['CBRE','CIGI','CSGP','CSR','FRPH','IRCP','JLL','KW','NTP','TCI'] # 10
        # Sector10
        # Code_list = ['ADSK','ANSS','CDNS','CRM','CTXS','INTU','PTC','SAP','SSNC','TYL'] # 10
        # Sector11
        # Code_list = ['AEP','DTE','DUK','ED','ES','NEE','PCG','SO','WFC','XEL'] #10

        ##########
        # Code_list = ['CDNS','CTXS','PTC','SAP','SSNC']
        # Code_list = ['ADSK','ANSS','CRM','INTU','TYL']
        #########################################
        # Code_list = file_name('./data/ISFD21-Arima')
        data_path = os.path.join('./data/ISFD21/ISFD-V1_11.csv')
        df = pd.read_csv(data_path)
        df.dropna(axis=0, how='any', inplace=True)
        df_group_data = pd.DataFrame()  # 存放group的Features
        df_label_data = pd.DataFrame()  # 存放features对应的label 由于底层是回归任务，因此对应的label是value形式
        for code in Code_list:
            df_code = df[(df['Code'] == code)]
            df_code_adj_close = df_code['Adj Close'].values.tolist()
            df_group_data[code] = df_code_adj_close
            df_code_label = df_code['Label'].values.tolist()
            df_label_data[code] = df_code_label

        df_group_data = pd.DataFrame(df_group_data.values.T, index=df_group_data.columns, columns=df_group_data.index)

        data = np.transpose([df_group_data.values])
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    # 2021-07-05 update standardScaler-start
    # standardScaler
    std = StandardScaler()
    x_stand = data.reshape(data.shape[0],data.shape[1])
    print('x_stand.shape:{}'.format(x_stand.shape))
    # x_stand_values = x_stand.values
    x_stand_values_std = std.fit_transform(x_stand)
    print('s_stand_values_std.mean:{}'.format(std.mean_))
    mean_list = std.mean_
    print(sum(mean_list)/10) # 这是原始用的mean_value  我们用股票全集的mean_value 来标准化数据？
    # 2021-07-05 update standardScaler-end

    # Normalization using Z-score method
    X = data
    print('X.shape:{}'.format(X.shape))
    # means = np.mean(X, axis=(0, 2))
    means = np.mean(X, axis=(0, 1))

    X = X - means.reshape(1, -1, 1)
    # stds = np.std(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 1))
    X = X / stds.reshape(1, -1, 1)

    means = means.reshape(1,1,1,1)
    stds = stds.reshape(1,1,1,1)
    print('means:{}'.format(means))
    print('stds:{}'.format(stds))
    return X,means,stds,data


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[0] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(X[i: i + num_timesteps_input,:,:])
        target.append(X[i + num_timesteps_input: j,:,0])
    print('features.shape:{}'.format(np.array(features).shape))
    print('target.shape:{}'.format(np.array(target).shape))
    # return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))
    return torch.tensor([np.array(feature) for feature in features],dtype=torch.float),torch.tensor([np.array(targe) for targe in target],dtype=torch.float)

def generate_dataset_classification(X, Labels, num_timesteps_input,operation,tmp_train,tmp_val,tmp_test):
    indices = [i for i in range(0,X.shape[0] - num_timesteps_input + 1)]
    if operation == 'train':
        # temp_length = 302
        temp_length = tmp_train
    elif operation == 'val':
        # temp_length = 101
        temp_length = tmp_val
    elif operation == 'test':
        # temp_length = 101
        temp_length = tmp_test
    indices_label = [i for i in range(0,temp_length - num_timesteps_input + 1)]
    features, target = [], []
    for i in indices:
        features.append(X[i: i + num_timesteps_input,:,:])
        # original
        # target.append(Labels[i + num_timesteps_input,:])
    for i in indices_label:
        target.append(Labels[i + num_timesteps_input-1,:,:])
    print('features.shape:{}'.format(np.array(features).shape))
    print('labels.shape:{}'.format(np.array(target).shape))
    return torch.tensor([np.array(feature) for feature in features],dtype=torch.float),torch.tensor([np.array(targe) for targe in target],dtype=torch.float)


def load_graphdata_channel_stp(Data_name,num_timesteps_input,num_timesteps_output, DEVICE, batch_size, shuffle=True):
    """
    Final Data prepare function for AASTGCN
    :return:
    """
    # # load data-ISFD21
    # X, means, stds, data = load_st_dataset(Data_name)

    # for NASDAQ single feature classification
    # X,Labels = load_ST_data_classification(data_path='./data/2013-01-01',market_name='NASDAQ',steps=1)
    
    # for NYSE single feature classification
    # X,Labels = load_ST_data_classification(data_path='./data/2013-01-01',market_name='NYSE',steps=1)
    
    # for NASDAQ multi feature classification
    # X,Labels = load_ST_data_classification_multi_feature(data_path='./data/2013-01-01',market_name='NASDAQ',steps=1)
    
    # for ACL18 multi feature classification
    # X,Labels = load_ST_data_classification_multi_feature_ACL18(data_path='./data/ACL18/preprocessed/')

    # for ACL18 multi 11 features classification
    _,Labels = load_ST_data_classification_multi_feature_ACL18(data_path='./data/ACL18/preprocessed/')
    X = load_ST_data_classification_multi_11features_ACL18(data_path='./data/ACL18/ourpped/')

    # for KDD17 multi 11 features classification
    # _,Labels = load_ST_data_classification_multi_feature_KDD17(data_path='./data/KDD17/raw/')
    # X = load_ST_data_classification_multi_11features_KDD17(data_path='./data/KDD17/preprocessed/')
    
    # no use
    # X = load_ST_data_classification_multi_11features_KDD17(data_path='./data/KDD17/latest_new_ourpped/')
    print('$$$$$$$$$ $$$$$$$$$$$$$$$$$$')
    print('X.shape:{}'.format(X.shape))
    print('labels.shape:{}'.format(Labels.shape))
    print('$$$$$$$$$ $$$$$$$$$$$$$$$$$$')
    # for ADGAT dataset
    # X,Labels = load_dataset()

    print('X.shape:{}'.format(X.shape)) # X.shape:(2516, 105, 1) (len,nodes,feature_dim)
    # print(X)

    ### 6:2:2 split method 2014/01/02-2015/03/16-2015/08/07
    split_line1 = int(X.shape[0] * 0.6)
    split_line2 = int(X.shape[0] * 0.8)
    # print('split_line1:{}'.format(split_line1))
    # print('split_line2:{}'.format(split_line2))
    
    ### Adv-LSTM split method (ACL18)
    # split_line1 = 398
    # split_line2 = 398 + 42

    ### Adv-LSTM split method (KDD17)
    # split_line1 = 2014
    # split_line2 = 2266


    ### AD-GAT split method
    # split_line1 = 590
    # split_line2 = 590 + 70
   
    tmp_length_train = split_line1
    tmp_length_val = split_line2 - split_line1
    tmp_length_test = X.shape[0] - split_line2
    print('train_length:{}'.format(tmp_length_train))
    print('val_length:{}'.format(tmp_length_val))
    print('test_length:{}'.format(tmp_length_test))
    # 应该是在这里进行数据的标准化处理。所以这个函数要接收一个standardscaler类，或者将standardscaler作为一个全局变量，处理其中数据
    train_original_data = X[:split_line1,:,:]
    val_original_data = X[split_line1:split_line2,:,:]
    test_original_data = X[split_line2:,:,:]

    Labels_train = Labels[:split_line1,:,:]
    Labels_val = Labels[split_line1:split_line2,:,:]
    Labels_test = Labels[split_line2:,:,:]
   # train valid test
    train_x, train_target = generate_dataset_classification(train_original_data,Labels_train,
                                                      num_timesteps_input=num_timesteps_input,operation='train', tmp_train = tmp_length_train, tmp_val = tmp_length_val, tmp_test = tmp_length_test)
    train_x = train_x.permute((0,2,3,1))
    ### train_target = train_target.permute((0,2,1))
    # train_target = np.expand_dims(train_target,axis=-1)
    print('trainx.shape:{}'.format(train_x.shape))
    print('train_target.shape:{}'.format(train_target.shape))

    val_x, val_target = generate_dataset_classification(val_original_data,Labels_val,
                                                      num_timesteps_input=num_timesteps_input,operation='val', tmp_train = tmp_length_train, tmp_val = tmp_length_val, tmp_test = tmp_length_test)
    val_x =  val_x.permute((0,2,3,1))
    ### val_target = val_target.permute((0,2,1))
    # val_target = np.expand_dims(val_target,axis=-1)
    print('val_x.shape:{}'.format(val_x.shape))
    print('val_target.shape:{}'.format(val_target.shape))
    print('val_x.shape:{}'.format(val_x.shape))
    print('val_target.shape:{}'.format(val_target.shape))

    test_x, test_target = generate_dataset_classification(test_original_data,Labels_test,
                                         num_timesteps_input=num_timesteps_input,operation='test', tmp_train = tmp_length_train, tmp_val = tmp_length_val, tmp_test = tmp_length_test)
    test_x = test_x.permute((0, 2, 3, 1))
    ### test_target = test_target.permute((0, 2, 1))
    # test_target = np.expand_dims(test_target,axis=-1)
    print('test_x.shape:{}'.format(test_x.shape))
    print('test_target.shape:{}'.format(test_target.shape))
    print('test_x.shape:{}'.format(test_x.shape))
    print('test_target.shape:{}'.format(test_target.shape))

    # means stds for normalize and recover the data
    # print('means.shape:{}'.format(means.shape))
    # print('stds.shape:{}'.format(stds.shape))

    # torch-float to avoid data type error
    train_x = np.array(train_x)
    train_target = np.array(train_target)

    val_x = np.array(val_x)
    val_target = np.array(val_target)

    test_x = np.array(test_x)
    test_target = np.array(test_target)
    print('test_target.shape:{}'.format(test_target.shape))
    print(test_target[-10:,0,:])

    # # load ISFD21-adj_matrix
    # adj_mx = np.load('./data/ISFD21_adj.npy') # static_adj_matrix with history price data

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    # return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, means, stds
    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor

def load_graphdata_channel_stp_standard(Data_name,num_timesteps_input,num_timesteps_output, DEVICE, batch_size, shuffle=True,scaler_method=StandardScaler()):
    """
    Final Data prepare function for AASTGCN
    :return:
    """
    # step 1 载入原始数据
    # # load data-ISFD21
    X, means, stds, data = load_st_dataset(Data_name) # 这里X是原始函数的标准化方法，data是标准化之前的数据

    print('data.shape:{}'.format(data.shape)) # X.shape:(2516, 105, 1) (len,nodes,feature_dim)
    print(data)
    # step 2 划分train/val/test
    split_line1 = int(X.shape[0] * 0.6)
    split_line2 = int(X.shape[0] * 0.8)
    # train_original_data = X[:split_line1,:,:]
    # val_original_data = X[split_line1:split_line2,:,:]
    # test_original_data = X[split_line2:,:,:]
    train_original_data = data[:split_line1,:,:]
    val_original_data = data[split_line1:split_line2,:,:]
    test_original_data = data[split_line2:,:,:]
    # step 3 对训练集进行fit_transform()
    std = scaler_method
    # std = StandardScaler()
    train_data = train_original_data.reshape(train_original_data.shape[0], train_original_data.shape[1])
    train_stand_data = std.fit_transform(train_data)
    # step 4 经过标准化的train_data 还原成原始维度(train_original_data)
    print('train_stand_data.shape:{}'.format(train_stand_data.shape))
    if len(train_stand_data.shape) == 2:
        train_stand_data = np.expand_dims(train_stand_data, axis=-1)
    print('train_stand_data.shape:{}'.format(train_stand_data.shape))

    # step 5 对验证集、测试集进行transform(),并还原成原始维度（val/test_original_data）
    val_data = val_original_data.reshape(val_original_data.shape[0], val_original_data.shape[1])
    val_stand_data = std.transform(val_data)
    if len(val_stand_data.shape) == 2:
        val_stand_data = np.expand_dims(val_stand_data, axis=-1)
    print('val_stand_data.shape:{}'.format(val_stand_data.shape))

    test_data = test_original_data.reshape(test_original_data.shape[0], test_original_data.shape[1])
    test_stand_data = std.transform(test_data)
    if len(test_stand_data.shape) == 2:
        test_stand_data = np.expand_dims(test_stand_data, axis=-1)
    print('test_stand_data.shape:{}'.format(test_stand_data.shape))
   # step 用train/val/test_stand_data 替换原来的trian/val/test_original_data,生成对应格式数据
   # train valid test
   #  train_x, train_target = generate_dataset(train_original_data,
   #                                                    num_timesteps_input=num_timesteps_input,
   #                                                    num_timesteps_output=num_timesteps_output)
    train_x, train_target = generate_dataset(train_stand_data,
                                                      num_timesteps_input=num_timesteps_input,
                                                      num_timesteps_output=num_timesteps_output)
    train_x = train_x.permute((0,2,3,1))
    train_target = train_target.permute((0,2,1))
    print('trainx.shape:{}'.format(train_x.shape))
    print('train_target.shape:{}'.format(train_target.shape))

    val_x, val_target = generate_dataset(val_stand_data,
                                                      num_timesteps_input=num_timesteps_input,
                                                      num_timesteps_output=num_timesteps_output)
    val_x =  val_x.permute((0,2,3,1))
    val_target = val_target.permute((0,2,1))
    print('val_x.shape:{}'.format(val_x.shape))
    print('val_target.shape:{}'.format(val_target.shape))

    test_x, test_target = generate_dataset(test_stand_data,
                                         num_timesteps_input=num_timesteps_input,
                                         num_timesteps_output=num_timesteps_output)
    test_x = test_x.permute((0, 2, 3, 1))
    test_target = test_target.permute((0, 2, 1))
    print('test_x.shape:{}'.format(test_x.shape))
    print('test_x:')
    print(test_x)
    print('test_target.shape:{}'.format(test_target.shape))


    # means stds for normalize and recover the data
    # print('means.shape:{}'.format(means.shape))
    # print('stds.shape:{}'.format(stds.shape))

    # torch-float to avoid data type error
    train_x = np.array(train_x)
    train_target = np.array(train_target)

    val_x = np.array(val_x)
    val_target = np.array(val_target)

    test_x = np.array(test_x)
    test_target = np.array(test_target)


    # # load ISFD21-adj_matrix
    # adj_mx = np.load('./data/ISFD21_adj.npy') # static_adj_matrix with history price data

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor

# load_graphdata_channel_stp(num_timesteps_input=12, num_timesteps_output=12,DEVICE=DEVICE, batch_size=64)

# load_st_dataset_standard("ISFD21")
# 现在load_graph_channel_stp_standard 所得到的train/val/test数据的标准化方式就是正常逻辑的了
# load_graphdata_channel_stp_standard("ISFD21",12,12,0,32,std_method)

##########################################################################

def filter_industry_tickers(market_name):
    '''
    filter out the stocks with missing values
    filter out the stocks with no industry attributes
    filter out the industry without any industry attribute
    filter out the industry attribute with only 1 stock
    :param market_name: NASDAQ or NYSE
    :return: dict->{industry:{tickers list}}
    '''
    # filter out the stocks with missing values
    stock_id_file_name = './data/'+ market_name + "_tickers_qualify_dr-0.98_min-5_smooth.csv"
    stock_id_list = pd.read_csv(stock_id_file_name, header=None).values.flatten()
    industry_ticker_file = './data/relation/relation/sector_industry/' + market_name +'_industry_ticker.json'
    missing_values_tickers_list = []
    mask = 1
    for index, ticker in enumerate(stock_id_list):
        # print(index, ticker)
        single_EOD = np.genfromtxt(
            os.path.join('./data/2013-01-01', market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ' or market_name == 'NYSE':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
            # print('single EOD data shape:', single_EOD.shape)
            for row in range(single_EOD.shape[0]):  # for each trading day
                if abs(single_EOD[row][-1] + 1234) < 1e-8:  # if the close price is none
                    mask = 0
        if mask == 0:
            missing_values_tickers_list.append(ticker)
        mask = 1
    # filter out the industry attribute with only 1 stock
    # filter out the stocks with no industry attributes
    # filter out the industry without any industry attribute
    # filter out the industry attribute with only 1 stock according to miss_values_tickers_list updated
    with open(industry_ticker_file, 'r') as read_file:
        industry_ticker = json.load(read_file)
        # print('len industry ticker:{}'.format(len(list(industry_ticker.keys()))))  # 113-NASDAQ & 130-NYSE
        for k in list(industry_ticker.keys()):
            # if k == 'n/a' or len(industry_ticker[k]) < 2:
            if len(industry_ticker[k]) < 2:
                # print('round 1 err industries:{}'.format(k))
                industry_ticker.pop(k)
        # print('len industry ticker:{}'.format(len(list(industry_ticker.keys()))))  # 95-NASDAQ & 106-NYSE
        for k in list(industry_ticker.keys()):
            temp_values = []
            for val in industry_ticker[k]:
                if val not in missing_values_tickers_list:
                    temp_values.append(val)
            industry_ticker[k] = temp_values
        for k in list(industry_ticker.keys()):
            if len(industry_ticker[k]) < 2:
                # print('round 2 err industries:{}'.format(k))
                industry_ticker.pop(k)
        # print('valid industry-ticker:{}'.format(len(list(industry_ticker.keys()))))  # 90-NASDAQ & 106-NYSE
        # print(industry_ticker)
        read_file.close()
        return industry_ticker

def load_ST_data(data_path, market_name, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    industry_ticker = filter_industry_tickers(market_name)
    tickers_list = [i for ii in industry_ticker.values() for i in ii]
    tickers = tickers_list
    for index, ticker in enumerate(tickers): # for each stock
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ' or market_name == 'NYSE':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            # original feature after norm: [time_index, 5-day, 10-day, 20-day, 30-day, close_price]
            print('single EOD data shape:', single_EOD.shape) # (1245, 6) (time_length, original_feature)

            # eod_data = (1026, 1245, 5) (stock_number, time_length, feature_number)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            # masks = (1026,1245)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            # ground_truth = (1026, 1245)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            # base_price = (1026, 1245)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]): # for each trading day
            if abs(single_EOD[row][-1] + 1234) < 1e-8: # if the close price is none
                masks[index][row] = 0.0 # the target stock's mask in that trading day is 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8: # if the close price is not none, the ground truth is ((close_price_t - close_price_t-1) / close_price_t-1)
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]): # for each feature
                if abs(single_EOD[row][col] + 1234) < 1e-8: # if the feature in that trading day is none
                    single_EOD[row][col] = 1.1 # set that feature equal to 1.1

        eod_data[index, :, :] = single_EOD[:, 1:] # set the stock eod_data as the latest single_EOD expect the time_index col
        base_price[index, :] = single_EOD[:, -1] # set the base_price, ie. the close_price
        X = base_price.T
        if len(base_price.shape) == 2:
            X = np.expand_dims(X, axis=-1)
    # return eod_data, masks, ground_truth, base_price
    return X

def load_ST_data_classification_err(data_path, market_name, steps=1):
    stock_close_price_data = load_ST_data(data_path=data_path, market_name=market_name, steps=steps)  # (1245,840,1)
    stock_close_price_data = np.squeeze(stock_close_price_data).T  # (840,1245) (stock_num,close_price)
    print(stock_close_price_data.shape)
    print(stock_close_price_data)
    # 根据stock_close_price_data 生成标签，实际上就是做矩阵的差分
    label_matrix = np.zeros([stock_close_price_data.shape[0], stock_close_price_data.shape[1] - 1])
    print(label_matrix.shape)
    for index in range(stock_close_price_data.shape[0]):
        for col in range(label_matrix.shape[1]):
            label_matrix[index][col] = stock_close_price_data[index][col + 1] - stock_close_price_data[index][col]
            if label_matrix[index][col] >= 0:
                label_matrix[index][col] = 1.0
            elif label_matrix[index][col] < 0:
                label_matrix[index][col] = 0.0
    # 生成好标签后，再对应去掉最后一个交易日的数据因为没有labels
    stock_close_price_data = stock_close_price_data[:, :-1]
    # 接下来，如果取一段时间的数据，那我们只需要拿到数据尾部那行的labels就可以了
    print(stock_close_price_data.shape)
    print(label_matrix)
    stock_close_price_data = stock_close_price_data.T
    stock_close_price_data = np.expand_dims(stock_close_price_data,axis=-1)
    label_matrix = label_matrix.T
    return stock_close_price_data, label_matrix

def load_ST_data_classification_err2(data_path, market_name, steps=1):
    stock_close_price_data = load_ST_data(data_path=data_path, market_name=market_name, steps=steps)  # (1245,840,1)
    stock_close_price_data = np.squeeze(stock_close_price_data).T  # (840,1245) (stock_num,close_price)
    print(stock_close_price_data.shape)
    print(stock_close_price_data)
    # 根据stock_close_price_data 生成标签，实际上就是做矩阵的差分
    label_matrix = np.zeros([stock_close_price_data.shape[0], stock_close_price_data.shape[1] - 1,2])
    print(label_matrix.shape)
    for index in range(stock_close_price_data.shape[0]):
        for col in range(label_matrix.shape[1]):
            temp = stock_close_price_data[index][col + 1] - stock_close_price_data[index][col]
            if temp >= 0:
                label_matrix[index][:] =[1.0,0.0]
            else:
                label_matrix[index][:] = [0.0,1.0]
    # 生成好标签后，再对应去掉最后一个交易日的数据因为没有labels
    stock_close_price_data = stock_close_price_data[:, :-1]
    # 接下来，如果取一段时间的数据，那我们只需要拿到数据尾部那行的labels就可以了
    # print(stock_close_price_data.shape)
    # print(label_matrix)
    stock_close_price_data = stock_close_price_data.T
    stock_close_price_data = np.expand_dims(stock_close_price_data,axis=-1)
    label_matrix = np.transpose(label_matrix,[1,0,2])
    return stock_close_price_data, label_matrix
##########################################################################
def load_ST_data_classification(data_path, market_name, steps=1):
    positive_sample = 0
    negative_sample = 0
    stock_close_price_data = load_ST_data(data_path=data_path, market_name=market_name, steps=steps)  # (1245,840,1)
    stock_close_price_data = np.squeeze(stock_close_price_data).T  # (840,1245) (stock_num,close_price)
    print(stock_close_price_data.shape)
    print(stock_close_price_data)
    # 根据stock_close_price_data 生成标签，实际上就是做矩阵的差分
    # label_matrix = np.zeros([stock_close_price_data.shape[0], stock_close_price_data.shape[1] - 1,2])
    label_matrix = np.zeros([stock_close_price_data.shape[1]-1, stock_close_price_data.shape[0],2]) # 1244,840
   
    for index in range(label_matrix.shape[1]):
        for col in range(label_matrix.shape[0]-1):
            temp = stock_close_price_data[index][col + 1] - stock_close_price_data[index][col]
            if temp >= 0:
                label_matrix[col,index,:] =[1.0,0.0]
                positive_sample += 1
            else:
                label_matrix[col,index,] = [0.0,1.0]
                negative_sample += 1
    '''# 生成好标签后，再对应去掉最后一个交易日的数据因为没有labels
    stock_close_price_data = stock_close_price_data[:, :-1]
    # 接下来，如果取一段时间的数据，那我们只需要拿到数据尾部那行的labels就可以了
    label_matrix = np.zeros([stock_close_price_data.shape[0], stock_close_price_data.shape[1] - 1,2])
    print(label_matrix.shape)
    for index in range(stock_close_price_data.shape[0]):
        for col in range(label_matrix.shape[0]):
            temp = stock_close_price_data[index][col + 1] - stock_close_price_data[index][col]
            if temp >= 0:
                label_matrix[index,col,:] =[1.0,0.0]
            else:
                label_matrix[index,col,] = [0.0,1.0]'''
    # 生成好标签后，再对应去掉最后一个交易日的数据因为没有labels
    stock_close_price_data = stock_close_price_data[:, :-1]
    # 接下来，如果取一段时间的数据，那我们只需要拿到数据尾部那行的labels就可以了
    # print(stock_close_price_data.shape)
    # print(label_matrix)
    stock_close_price_data = stock_close_price_data.T
    stock_close_price_data = np.expand_dims(stock_close_price_data,axis=-1)
    # label_matrix = np.transpose(label_matrix,[1,0,2])
    # label_matrix = label_matrix.reshape([label_matrix.shape[1],label_matrix.shape[0],label_matrix.shape[-1]])
    print('**********************************')
    print('positive samples:{} negative samples:{}'.format(positive_sample,negative_sample))
    print('**********************************')
    return stock_close_price_data, label_matrix
##########################################################################


######################multi_feature_classification########################

def load_ST_data_multi_feature(data_path, market_name, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    industry_ticker = filter_industry_tickers(market_name)
    tickers_list = [i for ii in industry_ticker.values() for i in ii]
    tickers = tickers_list
    for index, ticker in enumerate(tickers): # for each stock
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ' or market_name == 'NYSE':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            # original feature after norm: [time_index, 5-day, 10-day, 20-day, 30-day, close_price]
            print('single EOD data shape:', single_EOD.shape) # (1245, 6) (time_length, original_feature)

            # eod_data = (1026, 1245, 5) (stock_number, time_length, feature_number)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            # masks = (1026,1245)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            # ground_truth = (1026, 1245)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            # base_price = (1026, 1245)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]): # for each trading day
            if abs(single_EOD[row][-1] + 1234) < 1e-8: # if the close price is none
                masks[index][row] = 0.0 # the target stock's mask in that trading day is 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8: # if the close price is not none, the ground truth is ((close_price_t - close_price_t-1) / close_price_t-1)
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]): # for each feature
                if abs(single_EOD[row][col] + 1234) < 1e-8: # if the feature in that trading day is none
                    single_EOD[row][col] = 1.1 # set that feature equal to 1.1

        eod_data[index, :, :] = single_EOD[:, 1:] # set the stock eod_data as the latest single_EOD expect the time_index col
        base_price[index, :] = single_EOD[:, -1] # set the base_price, ie. the close_price
        X = base_price.T
        if len(base_price.shape) == 2:
            X = np.expand_dims(X, axis=-1)
    # print('eod_data.shape:{}'.format(eod_data.shape))
    # return eod_data, masks, ground_truth, base_price
    return base_price,eod_data

def load_ST_data_classification_multi_feature(data_path, market_name, steps=1):
    stock_close_price_data,eod_data = load_ST_data_multi_feature(data_path=data_path, market_name=market_name, steps=steps)  # (1245,840,1)
    stock_close_price_data = np.squeeze(stock_close_price_data)  # (840,1245) (stock_num,close_price)
    print('stock_close_price_data.shape:{}'.format(stock_close_price_data.shape))
    # print(stock_close_price_data)
    # 根据stock_close_price_data 生成标签，实际上就是做矩阵的差分
    label_matrix = np.zeros([stock_close_price_data.shape[0], stock_close_price_data.shape[1] - 1,2])
    # print(label_matrix.shape)
    for index in range(stock_close_price_data.shape[0]):
        for col in range(label_matrix.shape[1]):
            temp = stock_close_price_data[index][col + 1] - stock_close_price_data[index][col]
            if temp >= 0:
                # label_matrix[index][:] =[1.0,0.0]
                label_matrix[index,col,:] =[1.0,0.0]
            else:
                # label_matrix[index][:] = [0.0,1.0]
                label_matrix[index,col,:] = [0.0,1.0]
    # 生成好标签后，再对应去掉最后一个交易日的数据因为没有labels
    stock_close_price_data = stock_close_price_data[:, :-1]
    # 接下来，如果取一段时间的数据，那我们只需要拿到数据尾部那行的labels就可以了
    # print(stock_close_price_data.shape)
    # print(label_matrix)
    stock_close_price_data = stock_close_price_data.T
    stock_close_price_data = np.expand_dims(stock_close_price_data,axis=-1)
    # original_label_matrix = np.transpose(label_matrix,[1,0,2])
    label_matrix = label_matrix.reshape([label_matrix.shape[1],label_matrix.shape[0],label_matrix.shape[-1]])
    eod_data = eod_data[:,:-1,:]
    eod_data = eod_data.reshape([eod_data.shape[1],eod_data.shape[0],eod_data.shape[-1]])
    return eod_data, label_matrix
######################multi_feature_classification########################

#################################ACL18####################################
# 获取stock_id （文件名）
def files_name(path):
    filesname_list = []
    for i in range(len(path)):
        (filepath, tempfilename) = os.path.split(path[i])
        (filesname, extension) = os.path.splitext(tempfilename)
        filesname_list.append(filesname)
    # print("stock_id list:",filesname_list)
    return filesname_list

# 遍历文件夹，获取当前文件夹下文件所有路径的列表
def dir_name(path):
    file_list = os.listdir(path)
    file_name_list = []
    for i in range(len(file_list)):
        file_name = path + '/' + file_list[i]
        # print(file_name)
        file_name_list.append(file_name)
    return file_name_list

def load_ST_data_multi_feature_ACL18(data_path):
    # 获取文件路径列表
    file_path = dir_name(data_path)
    # 获取初始无后缀文件名
    stock_id_list = files_name(file_path)
    clear_stock_id_list = []
    for s in stock_id_list:
        if s not in ['AGFS', 'BABA', 'GMRE']:
            clear_stock_id_list.append(s)
    print(len(clear_stock_id_list))
    
    # industry-test
    ### Basic Matierials
    # clear_stock_id_list = ['XOM','RDS-B','PTR','CVX','TOT','BP','BHP','SNP','SLB','BBL']
    ### Consumer Goods
    # clear_stock_id_list = ['AAPL','PG','BUD','KO','PM','TM','PEP','UN','UL','MO'] 
    ### Healthcare
    # clear_stock_id_list = ['JNJ','PFE','NVS','UNH','MRK','AMGN','MDT','ABBV','SNY','CELG']
    ### Services #9
    # clear_stock_id_list = ['AMZN','WMT','CMCSA','HD','DIS','MCD','CHTR','UPS','PCLN']
    ### Utilities
    # clear_stock_id_list = ['NEE','DUK','D','SO','NGG','AEP','PCG','EXC','SRE','PPL']
    ### Conglomerates #6
    # clear_stock_id_list = ['IEP','HRG','CODI','REX','SPLP','PICO']
    # clear_stock_id_list = ['IEP','HRG','CODI','PICO']  
    # clear_stock_id_list = ['REX','SPLP']
    # clear_stock_id_list = ['CODI','HRG']
    # clear_stock_id_list = ['IEP','PICO']
    # clear_stock_id_list = ['HRG','CODI','REX','SPLP']
    ### Financial
    # clear_stock_id_list = ['BCH','BSAC','BRK-A','JPM','WFC','BAC','V','C','HSBC','MA']
    ### Industrial Goods
    # clear_stock_id_list = ['GE','MMM','BA','HON','UTX','LMT','CAT','GD','DHR','ABB']
    ### Technology
    clear_stock_id_list = ['GOOG','MSFT','FB','T','CHL','ORCL','TSM','VZ','INTC','CSCO']
    clear_stock_id_list.sort()
    print(clear_stock_id_list)
    # eod_data = []
    for index, ticker in enumerate(clear_stock_id_list):
        if ticker not in ['AGFS', 'BABA', 'GMRE']:
            # print('index:{} ticker:{}'.format(index, ticker))
            single_eod_data = pd.read_csv(data_path + ticker + '.txt', header=None, delimiter='\t')
            single_eod_data = single_eod_data[single_eod_data[0] >= '2014-01-01']
            single_eod_data = single_eod_data[single_eod_data[0] <= '2016-01-04']
            single_eod_data = single_eod_data.sort_values(
                by=0).values  # date,movement percent,open,high,low,close,volume
            single_eod_data = single_eod_data[:, 1:-1]  # movement_percent,open,high,low,close
            # print('single EOD data shape:',single_eod_data.shape)  # (505,7) date,movement percent,open,high,low,close,volume
            
            ####################################
            raw_eod_data = pd.read_csv('./data/ACL18/raw/' + ticker + '.csv', delimiter=',')
            raw_eod_data = raw_eod_data[raw_eod_data['Date'] >= '2014-01-01']
            raw_eod_data = raw_eod_data[raw_eod_data['Date'] <= '2016-01-04']
            raw_eod_data = raw_eod_data.sort_values(by='Date').values  # date,movement percent,open,high,low,close,volume
            raw_eod_data = raw_eod_data[:, 1:-1]  # movement_percent,open,high,low,close
            ###################################
            
            if index == 0:
                # print(single_eod_data)
                # eod_data = (1026, 1245, 5) (stock_number, time_length, feature_number)
                eod_data = np.zeros([len(clear_stock_id_list), single_eod_data.shape[0],
                                     single_eod_data.shape[1]], dtype=np.float32)

                # base_price = (1026, 1245)
                base_price = np.zeros([len(clear_stock_id_list), single_eod_data.shape[0]],
                                      dtype=np.float32)
                # print('eod_data shape:',eod_data.shape)
                # print('base_price shape:',base_price.shape)
            
            #####################################
                raw_base_price = np.zeros([len(clear_stock_id_list), raw_eod_data.shape[0]],dtype=np.float32)
            #################################################
            eod_data[index, :, :] = single_eod_data[:, :]
            base_price[index, :] = single_eod_data[:, -1]

            raw_base_price[index,:] = raw_eod_data[:,-1]
    return base_price,eod_data,raw_base_price

def load_ST_data_classification_multi_feature_ACL18(data_path):
    positive_sample = 0
    negative_sample = 0
    constant_sample = 0
    stock_close_price_data, eod_data,raw_base_price = load_ST_data_multi_feature_ACL18(data_path=data_path)  # (1245,840,1)
    
    # stock_close_price_data = np.squeeze(stock_close_price_data)  # (840,1245) (stock_num,close_price)
    stock_close_price_data = np.squeeze(raw_base_price)
    
    
    print('stock_close_price_data.shape:{}'.format(stock_close_price_data.shape))
    # print(stock_close_price_data)
    # 根据stock_close_price_data 生成标签，实际上就是做矩阵的差分
    label_matrix = np.zeros([stock_close_price_data.shape[1] - 1, stock_close_price_data.shape[0], 2])
    # print(label_matrix.shape)
    # print('stock_close_price_data:{}'.format(stock_close_price_data[2]))
    for index in range(label_matrix.shape[1]):
        for col in range(label_matrix.shape[0]):
            temp = stock_close_price_data[index][col + 1] - stock_close_price_data[index][col]
            # if index == 1:
                # print('temp:{}'.format(temp))
            if temp >= 0:
                # label_matrix[index][:] =[1.0,0.0]
                label_matrix[col, index, :] = [1.0, 0.0]
                positive_sample += 1
            else:
                # label_matrix[index][:] = [0.0,1.0]
                label_matrix[col, index, :] = [0.0, 1.0]
                negative_sample += 1
    # 生成好标签后，再对应去掉最后一个交易日的数据因为没有labels
    stock_close_price_data = stock_close_price_data[:, :-1]
    # 接下来，如果取一段时间的数据，那我们只需要拿到数据尾部那行的labels就可以了
    # print(stock_close_price_data.shape)
    # print(label_matrix)
    stock_close_price_data = stock_close_price_data.T
    stock_close_price_data = np.expand_dims(stock_close_price_data, axis=-1)
    # do not transpose
    # original_label_matrix = np.transpose(label_matrix,[1,0,2])
    # label_matrix = label_matrix.reshape([label_matrix.shape[1], label_matrix.shape[0], label_matrix.shape[-1]])
    eod_data = eod_data[:, :-1, :]
    eod_data = eod_data.reshape([eod_data.shape[1], eod_data.shape[0], eod_data.shape[-1]])
    print('***********************************')
    print('positive samples:{} negative samples:{}'.format(positive_sample, negative_sample))
    print('***********************************')
    print('label matrix.shape:{}'.format(label_matrix.shape))
    print('AAPL label:')
    print(label_matrix[-10:,0,:])
    return eod_data, label_matrix
#################################ACL18####################################

#################################ACL18-11features####################################

def load_ST_data_classification_multi_11features_ACL18(data_path):
    file_path = dir_name(data_path)
    stock_id_list = files_name(file_path)
    clear_stock_id_list = []
    for s in stock_id_list:
        if s not in ['AGFS', 'BABA', 'GMRE']:
            clear_stock_id_list.append(s)
    print(len(clear_stock_id_list))
    # print(clear_stock_id_list)
    # industry-test
    ### Basic Matierials
    # clear_stock_id_list = ['XOM','RDS-B','PTR','CVX','TOT','BP','BHP','SNP','SLB','BBL']
    ### Consumer Goods
    # clear_stock_id_list = ['AAPL','PG','BUD','KO','PM','TM','PEP','UN','UL','MO'] 
    ### Healthcare
    # clear_stock_id_list = ['JNJ','PFE','NVS','UNH','MRK','AMGN','MDT','ABBV','SNY','CELG']
    ### Services #9
    # clear_stock_id_list = ['AMZN','WMT','CMCSA','HD','DIS','MCD','CHTR','UPS','PCLN']
    ### Utilities
    # clear_stock_id_list = ['NEE','DUK','D','SO','NGG','AEP','PCG','EXC','SRE','PPL']
    ### Conglomerates #6
    # clear_stock_id_list = ['IEP','HRG','CODI','REX','SPLP','PICO']
    # clear_stock_id_list = ['IEP','HRG','CODI','PICO']
    # clear_stock_id_list = ['REX','SPLP'] 
    # clear_stock_id_list = ['CODI','HRG']
    # clear_stock_id_list = ['IEP','PICO']
    # clear_stock_id_list = ['HRG','CODI','REX','SPLP']
    ### Financial
    # clear_stock_id_list = ['BCH','BSAC','BRK-A','JPM','WFC','BAC','V','C','HSBC','MA']
    ### Industrial Goods
    # clear_stock_id_list = ['GE','MMM','BA','HON','UTX','LMT','CAT','GD','DHR','ABB']
    ### Technology
    clear_stock_id_list = ['GOOG','MSFT','FB','T','CHL','ORCL','TSM','VZ','INTC','CSCO']
    clear_stock_id_list.sort()
    print(clear_stock_id_list)
    for index, ticker in enumerate(clear_stock_id_list):
        if ticker not in ['AGFS', 'BABA', 'GMRE']:
            # print('index:{} ticker:{}'.format(index, ticker))
            single_eod_data = pd.read_csv(data_path + ticker + '.csv', header=None, delimiter=',').values
            single_eod_data = single_eod_data[148:,:]
            if ticker == clear_stock_id_list[0]:
                # print('single EOD data shape:',single_eod_data.shape)
                # print(single_eod_data)
              eod_data = np.zeros([len(clear_stock_id_list), single_eod_data.shape[0],single_eod_data.shape[1]-2], dtype=np.float32)

            # base_price = (1026, 1245) # 这里就不需要base_price了，因为最后一行的标签缺失，我们要用raw data中的数据去补全最后一行的标签
              base_price = np.zeros([len(clear_stock_id_list), single_eod_data.shape[0]],dtype=np.float32)
            eod_data[index,:,:] = single_eod_data[:,:-2]
    eod_data = eod_data.reshape([eod_data.shape[1], eod_data.shape[0], eod_data.shape[-1]])
    print('eod_data.shape:{}'.format(eod_data.shape))
    return eod_data

#################################ACL18-11features####################################

##########ADGAT-Dataset-198 stocks-5features#########################################
def load_dataset():
    # original data with 730 trading dates
    '''with open('./data/ADGAT/x_numerical.pkl', 'rb') as handle:
        markets = pickle.load(handle)
        # markets = np.reshape(markets,[markets.shape[1],markets.shape[0],markets.shape[-1]])
    with open('./data/ADGAT/y_.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
        # y_load = np.reshape(y_load,[y_load.shape[1],y_load.shape[0]])
    '''
    '''
    # update data with 700 trading dates
    with open('./data/ADGAT/latest_x_numerical_aastgcn.pkl', 'rb') as handle:
        markets = pickle.load(handle)
        # markets = np.reshape(markets,[markets.shape[1],markets.shape[0],markets.shape[-1]])
    with open('./data/ADGAT/latest_y_numerical_aastgcn.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
        # y_load = np.reshape(y_load,[y_load.shape[1],y_load.shape[0]])
    '''
    '''
    # update data with close price labels 700 samples
    with open('./data/ADGAT/x_numerical.pkl', 'rb') as handle:
        markets = pickle.load(handle)
        # markets = np.reshape(markets,[markets.shape[1],markets.shape[0],markets.shape[-1]])
    with open('./data/ADGAT/close_y_aastgcn.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
        # y_load = np.reshape(y_load,[y_load.shape[1],y_load.shape[0]])
   '''
    '''
   
   # update data with close price labels 730 samples
    with open('./data/ADGAT/x_numerical.pkl', 'rb') as handle:
        markets = pickle.load(handle)
        # markets = np.reshape(markets,[markets.shape[1],markets.shape[0],markets.shape[-1]])
    with open('./data/ADGAT/close_y_ADGAT_730.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
        # y_load = np.reshape(y_load,[y_load.shape[1],y_load.shape[0]])
    '''
   # update data with close price labels 730 samples and 10 features
    with open('./data/ADGAT/x_numerical_aastgcn_10features_730.pkl', 'rb') as handle:
        markets = pickle.load(handle)
        # markets = np.reshape(markets,[markets.shape[1],markets.shape[0],markets.shape[-1]])
    with open('./data/ADGAT/close_y_ADGAT_730.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
        # y_load = np.reshape(y_load,[y_load.shape[1],y_load.shape[0]])
    '''
   # update data with close price labels 700 samples and 10 features
    with open('./data/ADGAT/x_numerical_aastgcn_10features_700.pkl', 'rb') as handle:
        markets = pickle.load(handle)
        # markets = np.reshape(markets,[markets.shape[1],markets.shape[0],markets.shape[-1]])
    with open('./data/ADGAT/close_y_aastgcn.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
        # y_load = np.reshape(y_load,[y_load.shape[1],y_load.shape[0]])'''
    markets = markets.astype(np.float64)
    x = torch.tensor(markets)
    x.to(torch.double)

    y = torch.tensor(y_load)
    y = (y>0).to(torch.long)
    x = np.array(x)
    y = np.array(y)
    Label_matrix = np.zeros([y_load.shape[0], y_load.shape[1], 2])
    positive_samples = 0
    negative_samples = 0
    for index in range(y_load.shape[0]):
        for col in range(y_load.shape[1]):
            # print(y_load[index][col])
            if y[index][col] == 1:
                Label_matrix[index, col, :] = [1.0, 0.0]
                positive_samples += 1
            else:
                Label_matrix[index, col, :] = [0.0, 1.0]
                negative_samples += 1
    print('***********************************')
    print('positive samples:{} negative samples:{}'.format(positive_samples, negative_samples))
    print('***********************************')
    return x, Label_matrix
##########ADGAT-Dataset-198 stocks-5features#########################################


##############################################################################KDD17
def data_transform(x):
    # 日期斜杠转横杠
    publishtime = x
    array = time.strptime(publishtime, u"%m/%d/%Y")
    publishTime = time.strftime("%Y-%m-%d", array)
    return publishTime

date_err_stocks = ['AAPL','BRK-B','CMCSA','D','DUK','GOOGL','KO','MA','RIO','VALE','VZ']

def load_ST_data_multi_feature_KDD17(data_path):
    # 获取文件路径列表
    file_path = dir_name(data_path)
    # 获取初始无后缀文件名
    stock_id_list = files_name(file_path)
    clear_stock_id_list = []
    for s in stock_id_list:
        clear_stock_id_list.append(s)
    clear_stock_id_list.sort()
    # print(len(clear_stock_id_list))
    # print(clear_stock_id_list)
    for index, ticker in enumerate(clear_stock_id_list):
        print('index:{} ticker:{}'.format(index, ticker))
        single_eod_data = pd.read_csv('./data/KDD17/raw/' + ticker + '.csv',delimiter=',')
        if ticker in date_err_stocks:
            # single_eod_data = single_eod_data[single_eod_data['Date'] >= '1/3/2007']
            # single_eod_data = single_eod_data[single_eod_data['Date'] <= '1/5/2016']
            single_eod_data['Date'] = single_eod_data.Date.apply(lambda x: data_transform(x))

        single_eod_data = single_eod_data[single_eod_data['Date'] >= '2007-01-03']
        # 延长一天取数据为了制作最后一天数据的标签。
        single_eod_data = single_eod_data[single_eod_data['Date'] <= '2016-12-30']
        single_eod_data = single_eod_data.sort_values(by='Date').values
        if ticker in date_err_stocks:
            # date_err_stocks 的后两列是冗余数据需要剔除,前29行无法生成对应的特征也需要删除
            single_eod_data = single_eod_data[29:, 1:-2]
        else:
            # normal stocks 的最后一列就是adj price 无需改动，前29行无妨生成对应的特征也需要删除
            single_eod_data = single_eod_data[29:, 1:]
        if index == 0:
            # (stock_number, time_length, feature_number)
            eod_data = np.zeros([len(clear_stock_id_list), single_eod_data.shape[0],
                                 6], dtype=np.float32)

            # base_price
            base_price = np.zeros([len(clear_stock_id_list), single_eod_data.shape[0]],
                                  dtype=np.float32)


        eod_data[index, :, :] = single_eod_data[:, :]
        base_price[index, :] = single_eod_data[:, -1]
        # print('eod_data shape:', eod_data.shape)
        # print('base_price shape:', base_price.shape)
        # print(base_price[0])
    return base_price, eod_data

def load_ST_data_classification_multi_feature_KDD17(data_path):
    positive_sample = 0
    negative_sample = 0
    stock_close_price_data, eod_data = load_ST_data_multi_feature_KDD17(data_path=data_path)  # (1245,840,1)
    stock_close_price_data = np.squeeze(stock_close_price_data)  # (840,1245) (stock_num,close_price)
    print('stock_close_price_data.shape:{}'.format(stock_close_price_data.shape))
    # print(stock_close_price_data)
    # 根据stock_close_price_data 生成标签，实际上就是做矩阵的差分，去掉最后一行数据因为没有对应标签
    # label_matrix = np.zeros([stock_close_price_data.shape[1]-1, stock_close_price_data.shape[0], 2])
    label_matrix = np.zeros([stock_close_price_data.shape[1]-1, stock_close_price_data.shape[0], 2])
    # print(label_matrix.shape)
    print('stock_close_price_data:{}'.format(stock_close_price_data[2]))
    for index in range(label_matrix.shape[1]):
        for col in range(label_matrix.shape[0] - 1):
            temp = stock_close_price_data[index][col + 1] - stock_close_price_data[index][col]
            # if index == 1:
                # print('temp:{}'.format(temp))
            if temp >= 0:
                # label_matrix[index][:] =[1.0,0.0]
                label_matrix[col, index, :] = [1.0, 0.0]
                positive_sample += 1
            else:
                # label_matrix[index][:] = [0.0,1.0]
                label_matrix[col, index, :] = [0.0, 1.0]
                negative_sample += 1
    # 生成好标签后，再对应去掉最后一个交易日的数据因为没有labels
    stock_close_price_data = stock_close_price_data[:, :-1]
    # 接下来，如果取一段时间的数据，那我们只需要拿到数据尾部那行的labels就可以了
    # print(stock_close_price_data.shape)
    # print(label_matrix)
    stock_close_price_data = stock_close_price_data.T
    stock_close_price_data = np.expand_dims(stock_close_price_data, axis=-1)
    # do not transpose
    # original_label_matrix = np.transpose(label_matrix,[1,0,2])
    # label_matrix = label_matrix.reshape([label_matrix.shape[1], label_matrix.shape[0], label_matrix.shape[-1]])
    eod_data = eod_data[:, :-1, :]
    eod_data = eod_data.reshape([eod_data.shape[1], eod_data.shape[0], eod_data.shape[-1]])
    print('***********************************')
    print('positive samples:{} negative samples:{}'.format(positive_sample, negative_sample))
    print('***********************************')
    # print(label_matrix[0:10,0,:])
    return eod_data, label_matrix

def load_ST_data_classification_multi_11features_KDD17(data_path):
    file_path = dir_name(data_path)
    stock_id_list = files_name(file_path)
    clear_stock_id_list = []
    for s in stock_id_list:
        if s not in ['AGFS', 'BABA', 'GMRE']:
            clear_stock_id_list.append(s)
    clear_stock_id_list.sort()
    print(clear_stock_id_list)
    for index, ticker in enumerate(clear_stock_id_list):
        if ticker not in ['AGFS', 'BABA', 'GMRE']:
            print('index:{} ticker:{}'.format(index, ticker))
            single_eod_data = pd.read_csv(data_path + ticker + '.csv', header=None, delimiter=',').values
            single_eod_data = single_eod_data[29:-1,:-2]
            if ticker == clear_stock_id_list[0]:
                print('single EOD data shape:',single_eod_data.shape) # 2488,11
                print(single_eod_data)
                eod_data = np.zeros([len(clear_stock_id_list), single_eod_data.shape[0],
                                     single_eod_data.shape[1]], dtype=np.float32)

                # base_price = (1026, 1245) # 这里就不需要base_price了，因为最后一行的标签缺失，我们要用raw data中的数据去补全最后一行的标签
                base_price = np.zeros([len(clear_stock_id_list), single_eod_data.shape[0]],
                                      dtype=np.float32)
            eod_data[index,:,:] = single_eod_data[:,:]
    eod_data = eod_data.reshape([eod_data.shape[1], eod_data.shape[0], eod_data.shape[-1]])
    print('eod_data.shape:{}'.format(eod_data.shape))
    return eod_data
##############################################################################KDD17
