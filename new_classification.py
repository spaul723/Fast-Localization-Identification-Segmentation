import torch
import torch.nn as nn
import numpy as np
import os
import math

def correct(a,b):
    for i in range(5):
        if a[i]!=b[i]:
            return False
    return True

# 欧几里得距离
def euclidean_distance(x, y):
    dist = 0.0
    for i in range(len(x)):
        dist += (x[i] - y[i]) ** 2
    return math.sqrt(dist)

# get the result of the classification model
def get_classification(nii_name, start, end):
    seq_list = []
    for i in range(0, 20):
        seq_level = []
        for j in range(5):
            seq_level.append(i + j)
        seq_list.append(seq_level)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_data = np.zeros((5, 1, 64, 64, 32), dtype="float32")

    index = 0
    for i in range(start, end):
        img = np.load("./results/npy/" + nii_name[9:-7] + "/" + nii_name[9:-7] + "_" + str(i).zfill(2) + ".npy")
        img = (img - img.min()) / (img.max() - img.min())
        img_data[index] = img
        index += 1

    # model = resnet_lstm(101, 24, 20, 2, 24, 1, device).to(device)
    model = torch.load("./models/272_3d_resnet.pth").to(device)

    soft = nn.Softmax(dim=1)
    model.eval()

    print("Data has been loaded successfully.")

    img_data = torch.from_numpy(img_data).to(device)
    outputs = model(img_data).view((5, 24))
    outputs = soft(outputs)
    outputs = torch.argmax(outputs, dim=1)

    sim = 100
    temp = 0
    pred_correct = []
    for i in range(20):
        temp = euclidean_distance(outputs, seq_list[i])
        if temp < sim:
            pred_correct = seq_list[i]
            sim = temp

    print(pred_correct)
    return pred_correct







# def get_classification():
#
#     seq_list = []
#     for i in range(0,20):
#         seq_level = []
#         for j in range(5):
#             seq_level.append(i+j)
#         seq_list.append(seq_level)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     CT_NUM = 100
#
#     dataset = np.zeros((CT_NUM, 5, 64, 64, 32), dtype="float32")
#
#     index = 0
#     for filename in os.listdir("./test_L"):
#         sub_index = 0
#         for sub in os.listdir("./test_L/" + filename):
#             img = np.load("./test_L/" + filename + "/" + sub)
#             img = (img - img.min()) / (img.max() - img.min())
#             dataset[index][sub_index] = img
#             sub_index += 1
#         index += 1
#
#     model = torch.load("./model/173_3d_resnet.pth").to(device)
#     model.eval()
#
#     print("Everything is ok ~")
#
#     count_all = 50
#     for img_index in range(count_all):
#         img_data = dataset[img_index].reshape((5, 1, 64, 64, 32))
#         img_data = torch.from_numpy(img_data).to(device)
#         outputs = model(img_data).view((5, 24))
#         soft = nn.Softmax(dim=1)
#         outputs = soft(outputs)
#         outputs = torch.argmax(outputs, dim=1)
#
#         sim = 100
#         temp = 0
#         pred_correct = []
#         for i in range(20):
#             temp = euclidean_distance(outputs, seq_list[i])
#             if temp < sim:
#                 pred_correct = seq_list[i]
#                 sim = temp
#
#     return pred_correct