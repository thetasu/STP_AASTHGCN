import numpy as np
import pandas as pd
import os
import time
raw_data_path = 'price_long_50/'
ourpped_data_path = 'ourpped/'
# new_ourpped_data_path = 'new_ourpped/'
new_ourpped_data_path = 'latest_new_ourpped/'

def data_transform(x):
    # 日期斜杠转横杠
    publishtime = x
    array = time.strptime(publishtime, u"%m/%d/%Y")
    publishTime = time.strftime("%Y-%m-%d", array)
    return publishTime

date_err_stocks = ['AAPL','BRK-B','CMCSA','D','DUK','GOOGL','KO','MA','RIO','VALE','VZ']


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

##############################################################################KDD17
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
        single_eod_data = pd.read_csv(data_path + ticker + '.csv',delimiter=',')
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
        print('eod_data shape:', eod_data.shape)
        print('base_price shape:', base_price.shape)
        print(base_price[0])
    return base_price, eod_data

def load_ST_data_classification_multi_feature_KDD17(data_path):
    positive_sample = 0
    negative_sample = 0
    stock_close_price_data, eod_data = load_ST_data_multi_feature_KDD17(data_path=data_path)  # (1245,840,1)
    stock_close_price_data = np.squeeze(stock_close_price_data)  # (840,1245) (stock_num,close_price)
    print('stock_close_price_data.shape:{}'.format(stock_close_price_data.shape))
    # print(stock_close_price_data)
    # 根据stock_close_price_data 生成标签，实际上就是做矩阵的差分，去掉最后一行数据因为没有对应标签
    label_matrix = np.zeros([stock_close_price_data.shape[1]-1, stock_close_price_data.shape[0]])
    # print(label_matrix.shape)
    print('stock_close_price_data:{}'.format(stock_close_price_data[2]))
    for index in range(label_matrix.shape[1]):
        for col in range(label_matrix.shape[0] - 1):
            temp = stock_close_price_data[index][col + 1] - stock_close_price_data[index][col]
            # if index == 1:
            #     print('temp:{}'.format(temp))
            if temp >= 0:
                # label_matrix[index][:] =[1.0,0.0]
                label_matrix[col, index] = 1.0
                positive_sample += 1
            else:
                # label_matrix[index][:] = [0.0,1.0]
                label_matrix[col, index] = -1.0
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
            if ticker == 'AAPL':
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



###### generate new_ourpped dataset ↓

_,label_matrix = load_ST_data_classification_multi_feature_KDD17(raw_data_path)
print('label_matrix.shape:')
print(label_matrix.shape)
# 获取文件路径列表
file_path = dir_name(ourpped_data_path)
# 获取初始无后缀文件名
stock_id_list = files_name(file_path)
clear_stock_id_list = []
for s in stock_id_list:
    if s not in ['AGFS', 'BABA', 'GMRE']:
        clear_stock_id_list.append(s)
print(len(clear_stock_id_list))
clear_stock_id_list.sort()
print(clear_stock_id_list)
for index, ticker in enumerate(clear_stock_id_list):
    if ticker not in ['AGFS', 'BABA', 'GMRE']:
        # if ticker == 'AAPL':
        # print('index:{} ticker:{}'.format(index, ticker))
        single_eod_data = pd.read_csv(ourpped_data_path + ticker + '.csv', header=None, delimiter=',').values
        print('----------------')
        print(single_eod_data.shape)
        # single_eod_data = np.around(single_eod_data,6)
        print(single_eod_data[29:-1,].shape)
        single_eod_data[29:-1,-2] = label_matrix[:,index].flatten()
        # single_eod_data = single_eod_data[:-1,:] # why delete? now the data is clear!
        single_eod_data = np.around(single_eod_data,6)
        np.savetxt(new_ourpped_data_path + ticker +'.csv',single_eod_data,delimiter=',')
        print('new data save success')

##### generate new_ourpped dataset ↑

