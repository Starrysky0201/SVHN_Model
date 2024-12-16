import numpy as np
import torch
import SVHNData
import pandas as pd
from train import SVHN_Model1
import matplotlib.pyplot as plt

def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    use_cuda = True
    if use_cuda:
        model = model.cuda()
    # TTA 次数
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



def test_model():
    test_loader = SVHNData.GetTest_SVHNData()
    model = SVHN_Model1()
    # 加载保存的最优模型
    model.load_state_dict(torch.load('model.pt'))
    test_predict_label = predict(test_loader, model, 1)
    print(test_predict_label.shape)
    # 获取预测标签
    test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
    test_predict_label = np.vstack([
        test_predict_label[:, :11].argmax(1),
        test_predict_label[:, 11:22].argmax(1),
        test_predict_label[:, 22:33].argmax(1),
        test_predict_label[:, 33:44].argmax(1),
        test_predict_label[:, 44:55].argmax(1),
    ]).T
    test_label_pred = []
    for x in test_predict_label:
        test_label_pred.append(''.join(map(str, x[x != 10])))
    # 保持预测结果
    df_submit = pd.read_csv('../prediction_result/mchar_sample_submit_A.csv')
    df_submit['file_code'] = test_label_pred
    df_submit.to_csv('../prediction_result/submit_resnet50.csv', index=None)

if __name__ == '__main__':
    test_model()