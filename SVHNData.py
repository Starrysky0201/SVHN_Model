import os,sys,glob,shutil,json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SVHNDateset(Dataset):
    def __init__(self, img_path, img_label, transform = None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, item):
        # 读入图片
        img = Image.open(self.img_path[item]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # 原始SVHN中类别10表示空字符
        lbl = np.array(self.img_label[item], dtype=int)
        # 标签长度小于5，则用类别10来填充
        lbl = list(lbl) + (5 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

transform = transforms.Compose([
    # 统一尺寸为 64x128
    transforms.Resize((64,128)),
    # 随机裁剪
    transforms.RandomCrop((60, 120)),
    # 随机颜色变换
    transforms.ColorJitter(0.2,0.2,0.2),
    # 加入随机旋转
    transforms.RandomRotation(5),
    # 将图片转换为Pytorch 的 Tensor
    transforms.ToTensor(),
    # 归一化
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def GetTrain_SVHNData():
    train_path = glob.glob('../tcdata/mchar_train/*png')
    train_path.sort()
    train_json = json.load(open('../tcdata/mchar_train.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    train_loader = torch.utils.data.DataLoader(
        SVHNDateset(train_path, train_label,
                    transforms.Compose([
                        transforms.Resize((64, 128)),
                        transforms.RandomCrop((60, 120)),
                        transforms.ColorJitter(0.3, 0.3, 0.2),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=True,
        num_workers=10,
    )
    return train_loader

def GetTest_SVHNData():
    test_path = glob.glob('../tcdata/mchar_test_a/*png')
    test_path.sort()
    test_label = [[1]] * len(test_path)

    test_loader = torch.utils.data.DataLoader(
        SVHNDateset(test_path, test_label,
                    transforms.Compose([
                        transforms.Resize((70, 140)),
                        # transforms.RandomCrop((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=False,
        num_workers=10,
    )
    return test_loader

def GetVal_SVHNData():
    val_path = glob.glob('../tcdata/mchar_val/*.png')
    val_path.sort()
    val_json = json.load(open('../tcdata/mchar_val.json'))
    val_label = [val_json[x]['label'] for x in val_json]
    val_loader = torch.utils.data.DataLoader(
        SVHNDateset(val_path, val_label,
                    transforms.Compose([
                        transforms.Resize((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=False,
        num_workers=10,
    )
    return val_loader

if __name__ == '__main__':
    a=1
