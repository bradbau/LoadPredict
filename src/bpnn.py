import torch
import torch.nn as nn	# 各种层类型的实现
import torch.nn.functional as F	# 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim	# 实现各种优化算法的包
from torchvision import datasets, transforms
import torchData


class Net_1(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.conv=nn.Sequential()

        self.conv.add_module("rnn", nn.RNN(input_size, hidden_size, output_size,False, False, False))
        self.conv.add_module("relu",nn.ReLU())

    def forward(self, x):
        conv_out = self.conv1(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out





def train(args, model, device, train_loader, optimizer, epoch): # 还可添加loss_func等参数
    model.train() # 必备，将模型设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader): # 从数据加载器迭代一个batch的数据
        data, target = data.to(device), target.to(device) # 将数据存储CPU或者GPU
        optimizer.zero_grad() # 清除所有优化的梯度
        output = model(data)  # 喂入数据并前向传播获取输出
        loss = F.nll_loss(output, target) # 调用损失函数计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval() # 必备，将模型设置为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 禁用梯度计算
        for data, target in test_loader: # 从数据加载器迭代一个batch的数据
            data, target = data.to(device), target.to(device) 
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # 统计预测正确个数

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




if __name__=="__main__":
    datapath = r"inputdata\drastic.txt"
    batch_size = 64
    nraws = 1000
    train_dataset = Dataset(datapath,nraws)
    train_iter = Data.DataLoader(dataset = train_dataset, batch_size = batch_size)
    print(type(train_iter))



    for epoch in range(20):
        







