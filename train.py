import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import SVHNData
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
from torchvision import models


class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # 利用resnet18进行特征提取
        feature_map = models.resnet18(pretrained=True)
        # 平均池化
        feature_map.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_map = nn.Sequential(*list(feature_map.children())[:-1])
        self.cnn = feature_map
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
    def forward(self, img):
        feat = self.cnn(img)
        # 将feature_map转化为分类器的输入 512
        feat = feat.view(feat.shape[0], -1)
        # 获取5个分类器的输出，分别对于5个定长字符
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5

def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            # 利用GPU进行训练
            input = input.cuda()
            target = target.cuda()
        c0, c1, c2, c3, c4 = model(input)
        loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3]) + \
               criterion(c4, target[:, 4])
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        train_loss.append(loss.item())
    return np.mean(train_loss)

def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []
    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            c0, c1, c2, c3, c4 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3]) + \
                   criterion(c4, target[:, 4])
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    # 预测模式
    model.eval()
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()
                c0, c1, c2, c3, c4 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

# 可视化损失
def draw_result(epochs, train_loss, test_loss,test_acc):
    img = plt.figure()
    plt.plot(range(1, epochs+1),train_loss, label = 'Train loss',color = 'r')
    plt.plot(range(1, epochs+1),test_loss, label = 'Test loss',color = 'b')
    plt.plot(range(1, epochs+1),test_acc, label = 'Test acc',color = 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('result')
    plt.legend()
    plt.show()
    # plt.savefig('loss.jpg',img)

if __name__ == '__main__':
    # 导入训练和验证数据
    train_loader = SVHNData.GetTrain_SVHNData()
    val_loader = SVHNData.GetVal_SVHNData()
    # 定义模型
    model = SVHN_Model1()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义Adam优化器，设置学习率为0.0005
    optimizer = torch.optim.Adam(model.parameters(), 0.0005)
    # 保持最好模型
    best_loss = 1000.0
    # 是否使用GPU
    use_cuda = True
    train_losss = []
    test_losss = []
    test_accs = []
    if use_cuda:
        model = model.cuda()
    for epoch in range(1):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion)
        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
            val_predict_label[:, 44:55].argmax(1),
        ]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))
        # 获取准确率
        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print('Val Acc', val_char_acc)
        train_losss.append(train_loss)
        test_losss.append(val_loss)
        test_accs.append(val_char_acc)
        # 记录下验证集精度并保持最好的模型
        # 防止过拟合
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './model.pt')
    draw_result(1, train_losss, test_losss, test_accs)