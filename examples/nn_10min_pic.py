import torch
from torch.utils.data import DataLoader

# torchvision 是一个与 PyTorch 深度学习框架配套的库，它提供了处理图像和视频的常用工具和预训练模型
from torchvision import transforms
from torchvision.datasets import MNIST

# matplotlib 是一个 Python 绘图库，广泛用于生成高质量的图形和图表。它提供了一个类似于 MATLAB 的绘图框架。
import matplotlib.pyplot as plt


# 定义Net类，其父类是 @torch.nn.Module
class Net(torch.nn.Module):
    # 初始化设置为四个全连接层
    def __init__(self):
        super().__init__()
        # 输入为28*28像素尺寸的图像
        self.fc1 = torch.nn.Linear(28*28, 64)
        # 2，3，层输入64个节点，输出64个节点
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        # 第四层输入64个节点，输出10个数字类别
        self.fc4 = torch.nn.Linear(64, 10)

    # 定义前向传播过程，@x 为图像输入
    def forward(self, x):
        # 每一层传播中先做全连接线性计算[self.fc*(x)]，然后激活函数调整[relu(f(x))]
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        # 输出层通过softmax做归一化(对数运算提高稳定性log_softmax(f(x)))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

# @is_train 是否是训练数据集加载# 输入64个节点，输出64个节点；函数返回数据加载器
def get_data_loader(is_train):
    # 定义张量[多维数组]数据转换类型@to_tensor
    to_tensor = transforms.Compose([transforms.ToTensor()])
    
    # 下载MNIST数据集
    data_set = MNIST("", is_train, transform=to_tensor, download=True)

    # @batch_size 一批处理的图片数量, @shuffle 是否随机打乱数据
    return DataLoader(data_set, batch_size=15, shuffle=True)

# 评估神经网络的识别正确率
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        # 从测试数据集@test_data 批次取出数据
        for (x, y) in test_data:
            # 计算神经网络的预测值，输出到@outputs
            outputs = net.forward(x.view(-1, 28*28))
            # 对outputs中的每个结果output进行检验
            for i, output in enumerate(outputs):
                # @output中最大概率[argmax]的结果是多少
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    # 导入训练和测试数据集
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    # 打印初始神经网络的预测结果[预期接近0.1]
    print("Initial accurycy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        # 神经网络训练的固定写法
        for (x, y) in train_data:
            # 初始化
            net.zero_grad()
            # 正向传播
            output = net.forward(x.view(-1, 28*28))
            # 计算差值[nll_loss是对数损失函数，对应log_softmax的对数运算]
            loss = torch.nn.functional.nll_loss(output, y)
            # 反向误差传播
            loss.backward()
            # 优化网络参数
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 随机抽取三张图像显示网络的预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("Zhu Tao prediction: " + str(int(predict)))
    plt.show()

if __name__ == "__main__":
    main()
