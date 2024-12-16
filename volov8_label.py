import os,glob,json,shutil

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

DATA_HOME = '../tcdata'


def convert(size, top,height,left,width,label):
    """
    # 转换为YOLOv8归一化格式 [class_id, x_center, y_center, width, height]
    :param size: 图像尺寸
    :param top: 左上角Y
    :param height:字符高度
    :param left:左上角C
    :param width:字符宽度
    :param label:字符编码
    :return: [class_id, x_center, y_center, width, height]
    """
    h,w,_ =size
    top = np.array(top)
    height = np.array(height)
    left = np.array(left)
    width = np.array(width)
    label = np.array(label)

    y_center = top + height/2.0
    x_center = left + width/2.0

    y_center = y_center/h
    height = height / h
    x_center = x_center / w
    width = width /w

    return np.array([label, x_center, y_center, width,height]).T


def create_yolov8():
    # 创建数据配置文件
    with open(os.path.join(DATA_HOME, 'tcdata.yaml'), 'w') as f:
        f.write(f"""
            path: F:\\AI-learn\\project\\tcdata
            train: images/train
            val: images/val
            
            nc: {10}
            name: {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
            """)
    # 创建训练，验证目录
    os.makedirs(os.path.join(DATA_HOME, 'images\\train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_HOME, 'images\\val'), exist_ok=True)
    os.makedirs(os.path.join(DATA_HOME, 'images\\test'), exist_ok=True)
    os.makedirs(os.path.join(DATA_HOME, 'labels\\train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_HOME, 'labels\\val'), exist_ok=True)
    os.makedirs(os.path.join(DATA_HOME, 'labels\\test'), exist_ok=True)

    train_img = os.path.join(DATA_HOME, 'mchar_train')
    val_img = os.path.join(DATA_HOME, 'mchar_val')
    test_img = os.path.join(DATA_HOME, 'mchar_test_a')
    # 读取json文件
    train_json = json.load(open('../tcdata/mchar_train.json'))
    val_json = json.load(open('../tcdata/mchar_val.json'))
    for key in train_json:
        img = cv2.imread(os.path.join(train_img, key))
        cv2.imwrite(os.path.join(DATA_HOME, 'images\\train',key),img)
        top = train_json[key]['top']
        height =train_json[key]['height']
        left=train_json[key]['left']
        width=train_json[key]['width']
        label=train_json[key]['label']
        boxes = convert(img.shape,top,height, left, width, label)
        # 创建对应的txt标注文件
        label_path = os.path.join(DATA_HOME,'labels\\train', os.path.splitext(key)[0] + '.txt')
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(' '.join(map(str, box)) + '\n')
    for key in val_json:
        img = cv2.imread(os.path.join(val_img, key))
        cv2.imwrite(os.path.join(DATA_HOME, 'images\\val',key),img)
        top = val_json[key]['top']
        height =val_json[key]['height']
        left=val_json[key]['left']
        width=val_json[key]['width']
        label=val_json[key]['label']

        boxes = convert(img.shape,top,height, left, width, label)
        # 创建对应的txt标注文件
        label_path = os.path.join(DATA_HOME,'labels\\val', os.path.splitext(key)[0] + '.txt')
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(' '.join(map(str, box)) + '\n')

def train_yolov8():
    """
    使用YOLOv8训练目标检测模型
    """
    model = YOLO('yolov8n.pt')
    # 训练
    results = model.train(
        data='F:\\AI-learn\\project\\tcdata\\tcdata.yaml',
        epochs=10,
        batch=16,
        imgsz=640,
        device='0',  # GPU设备
        plots=True  # 生成训练过程的性能图
    )
    model.save('yolov8_tc.pt')

def test_yolov8():
    """
    使用YOLOv8进行预测
    """
    model = YOLO('yolov8_tc.pt')
    test_path = 'F:\\AI-learn\\project\\tcdata\\mchar_test_a'

    metrics = model.predict(test_path,show=False)
    test_pred = []
    for metric in metrics:
        boxes = metric.boxes.xywh
        boxes_x = boxes[:,0]
        sorted_tensor, sorted_indices =boxes_x.sort(dim=0)
        label = metric.boxes.cls[sorted_indices]
        test_pred.append(''.join([str(i) for i in label.int().tolist()]))
    # 保存结果为csv文件
    df_submit = pd.read_csv('../prediction_result/mchar_sample_submit_A.csv')
    df_submit['file_code'] = test_pred
    df_submit.to_csv('../prediction_result/submit_volov8.csv', index=None)



if __name__ == '__main__':
    # create_yolov8()
    train_yolov8()
    test_yolov8()












