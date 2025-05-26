import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import time
import numpy as np
import random


# 设置随机种子
seed = 42
torch.manual_seed(seed)
random.seed(seed)

data = pd.read_csv("/data2/zmy/lorenz_dataset-3.csv")
# data.plot()
# plt.show()

# 输入3个变量，预测3个变量，搭建3个连接层，使用3个损失函数，再将其进行相加
# 1 制作数据集，做成[Batch_size,时间长度，每个时间点的特征为3]
# 预测未来48小时

data = data.iloc[:, [1, 2, 3]].values  # 获取数值 456 *3
p = 1000 # 预测未来2个月24个小时的数据
train_data = data[:-p]
test_data = data[-p:]
min_data = np.min(train_data, axis=0) #求了每个特征的最小值，即每列的最小值，3*1的一维数组
max_data = np.max(train_data, axis=0)
train_data_scaler = (train_data - min_data) / (max_data - min_data)


def get_x_y(data, step=7*24):
    # x : 168*3
    # y: 1 *3
    x_y = []
    for i in range(len(data) - step):
        x = data[i:i + step, :]
        y = np.expand_dims(data[i + step, :], 0)
        x_y.append([x, y])
    return x_y


def get_mini_batch(data, batch_size=16):
    # x: 16 * 168 *3
    # y :16 * 1 *3
    for i in range(0, len(data) - batch_size, batch_size):
        samples = data[i:i + batch_size]
        x, y = [], []
        for sample in samples:
            x.append(sample[0])
            y.append(sample[1])
        yield np.asarray(x), np.asarray(y)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        '''
        Args:
        - y_pred: 预测值, size=[batch_size]
        - y_true: 目标值, size=[batch_size]
        Returns:
        - loss: MAPE值
        '''
        epsilon = 1e-8
        diff = torch.abs((y_true - y_pred) / torch.clamp(y_true, min=epsilon))
        loss_mape = torch.mean(diff)*100
        return loss_mape
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        '''
        Args:
        - y_pred: 预测值, size=[batch_size]
        - y_true: 目标值, size=[batch_size]
        Returns:
        - loss: RMSE值
        '''
        criteron = torch.nn.MSELoss()
        mse_loss = criteron(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

class BiLSTM(nn.Module):  # 注意Module首字母需要大写
    def __init__(self, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.hidden_size = hidden_size  # 隐含层神经元数目 100
        self.num_layers = num_layers  # 层数 通常设置为2
        self.output_size = output_size  # 24 一次预测下24个时间步
        self.num_directions = 2  # 双向LSTM
        self.input_size = 3
        self.batch_size = batch_size
        # 初始化隐藏层数据
        self.hidden_cell = (
            torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size),
            torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size))

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,bidirectional=True)  # 将LSTM改为BiLSTM
        self.fc1 = nn.Linear(self.num_directions * self.hidden_size, self.output_size)  # 修改全连接层的输入维度
        self.fc2 = nn.Linear(self.num_directions * self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.num_directions * self.hidden_size, self.output_size)
    def forward(self, input):
        output, _ = self.lstm(torch.FloatTensor(input), self.hidden_cell)
        pred1, pred2, pred3 = self.fc1(output[:, -1, :]), self.fc2(output[:, -1, :]), self.fc3(output[:, -1, :])  # 修改预测值的维度处理方式
        pred = torch.stack([pred1, pred2, pred3], dim=2)

        return pred



if __name__ == '__main__':
    time_step = 168  # 用过去12个小时，预测未来第13个小时，7天预测1天

    train_x_y = get_x_y(train_data_scaler, step=time_step)
    random.shuffle(train_x_y)
    batch_size = 24
    hidden_size = 100
    num_layers = 2
    output_size = 1

    model = BiLSTM(hidden_size, num_layers, output_size, batch_size)  # 使用修改后的BiLSTM
    mape_loss_func=MAPELoss()
    loss_function = RMSELoss()
    r2_loss_func=nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器实例
    print(model)
    epochs = 2
    loss_list = []
    mape_loss_list=[]
    r2_loss_list=[]
    for i in range(epochs):
        start = time.time()
        loss_all = 0
        mape_loss_all = 0
        r2_loss_all = 0
        num = 0
        for seq_batch, label_batch in get_mini_batch(train_x_y, batch_size):
            optimizer.zero_grad()
            y_pred = model(seq_batch)
            loss = 0
            mape_loss=0
            r2_loss=0
            for j in range(3):
                loss += loss_function(y_pred[:,:, j], torch.FloatTensor(label_batch[:, :, j]))
                mape_loss += mape_loss_func(y_pred[:, :, j], torch.FloatTensor(label_batch[:, :, j]))
                r2_loss += 1-r2_loss_func(y_pred[:, :, j], torch.FloatTensor(label_batch[:, :, j]))/torch.var(torch.FloatTensor(label_batch[:,:,j]))
                
            loss /= 3
            mape_loss /=3
            r2_loss /=3

            loss.backward()  # 调用loss.backward()自动生成梯度，
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络
            loss_all += loss.item()
            mape_loss_all += mape_loss.item()
            r2_loss_all += r2_loss.item()
            num += 1
            # 查看模型训练的结果
        loss_all /= num
        mape_loss_all /= num
        r2_loss_all /= num
        r2_loss_list.append(r2_loss_all)
        loss_list.append(loss_all)
        mape_loss_list.append(mape_loss_all)
        print(f'epoch:{i:3} loss:{loss_all:10.8f}  mape_loss:{mape_loss_all:10.8f}  r2:{r2_loss_all:10.8f}  time:{time.time() - start:6}')
        

    plt.plot(list(range(len(loss_list))), loss_list, 'b-')
   # plt.show()
    plt.savefig('/data2/zmy/img/双向用户10和13、14变量训练LOSS.png')
    fig = plt.figure() # 创建新的Figure对象
    plt.plot(list(range(len(mape_loss_list))), mape_loss_list, 'r-')
    plt.savefig('/data2/zmy/img/双向用户10和13、14变量训练MAPELOSS.png')

    model.eval()
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(model.num_directions * model.num_layers, 1, hidden_size),
                             torch.zeros(model.num_directions * model.num_layers, 1, hidden_size))
        # 测试集
        total_test_loss = 0
        test_pred = np.zeros(shape=(p, 3)) #p=8760
        for i in range(len(test_data)): #24
            x = train_data_scaler[-time_step:, :]
            x1 = np.expand_dims(x, 0)
            test_y_pred_scalar = np.expand_dims(model(x1).cpu().squeeze().numpy(), 0)  # 预测的值0-1
            train_data_scaler = np.append(train_data_scaler, test_y_pred_scalar, axis=0)
            y = test_y_pred_scalar * (max_data - min_data) + min_data
            test_pred[i,:] = y

        df=pd.DataFrame({'prex':test_pred[:,0],
                         'prey':test_pred[:,1],
                         'prez':test_pred[:,2]})
        df.to_csv('/data2/zmy/img/洛伦兹BiLSTM—pred.csv')

        x_in = list(range(len(test_pred))) #变成了每个块24*3，#48*3，共很多个块
       # fig = plt.figure() # 创建新的Figure对象
       # plt.plot(x_in, test_data[:, 0], 'r')
       # plt.plot(x_in, test_pred[:, 0], 'b')
        
        fig = plt.figure() # 创建新的Figure对象
        plt.plot(x_in, test_data[:, 0], 'r')
        plt.plot(x_in, test_pred[:, 0], 'b')
        plt.legend([ "true-x",  "pred-y"])     
        plt.savefig('/data2/zmy/img/多变量用户x.png')
        
        fig = plt.figure() # 创建新的Figure对象
        plt.plot(x_in, test_data[:, 1], 'r')
        plt.plot(x_in, test_pred[:, 1], 'b')
        plt.legend([ "true-y",  "pred-y"])     
        plt.savefig('/data2/zmy/img/多变量用户y.png')
        
        fig = plt.figure() # 创建新的Figure对象
        plt.plot(x_in, test_data[:, 2], 'r')
        plt.plot(x_in, test_pred[:, 2], 'b')
        plt.legend([ "true-z",  "pred-z"])     
     
       # plt.show()
        plt.savefig('/data2/zmy/img/多变量用户z.png')