from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torchvision.transforms import Normalize

from dataset import DatasetFromFolder

# from BCNN_VGG16 import Net
from BCNN_ResNet101 import Net

root_dir = '/home/wcc/data/car_attribute/processedImages'

def get_dict():
    dicts = []
    dict_path = "dict"
    dict_files = os.listdir(os.path.join(root_dir, dict_path))
    dict_files.sort()
    for dict_file in dict_files:
        with open(os.path.join(root_dir, dict_path, dict_file), "r") as f:
            d = {}
            lines = f.readlines()
            for line in lines:
                d[int(line.split()[0])] = line.split()[1]
            dicts.append(d)

    return dicts

def get_test_set():
    test_dir = os.path.join(root_dir, "test")
    crop_size = (224, 224)
    return DatasetFromFolder(test_dir, input_transform=Compose([Resize(crop_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

def batch_test():
    cm0 = np.zeros((model.classifier.fc1_1.out_features, model.classifier.fc1_1.out_features), int)
    cm1 = np.zeros((model.classifier.fc2_1.out_features, model.classifier.fc2_1.out_features), int)
    cm2 = np.zeros((model.classifier.fc3_1.out_features, model.classifier.fc3_1.out_features), int)
    cm3 = np.zeros((model.classifier.fc4_1.out_features, model.classifier.fc4_1.out_features), int)
    cm4 = np.zeros((model.classifier.fc5_1.out_features, model.classifier.fc5_1.out_features), int)
    cm5 = np.zeros((model.classifier.fc6_1.out_features, model.classifier.fc6_1.out_features), int)
    cm = [cm0, cm1, cm2, cm3, cm4, cm5]

    print('===> Loading datasets')
    test_set = get_test_set()
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    with torch.no_grad():
        # count = 0
        for batch in test_data_loader:
            input = batch[0].to(device)

            prediction = model(input)

            pred0 = prediction[0].max(1, keepdim=True)[1]
            pred1 = prediction[1].max(1, keepdim=True)[1]
            pred2 = prediction[2].max(1, keepdim=True)[1]
            pred3 = prediction[3].max(1, keepdim=True)[1]
            pred4 = prediction[4].max(1, keepdim=True)[1]
            pred5 = prediction[5].max(1, keepdim=True)[1]
            pred = [int(pred0[0][0]), int(pred1[0][0]), int(pred2[0][0]), int(pred3[0][0]), int(pred4[0][0]), int(pred5[0][0])]
            # print(int(pred0[0][0]), int(pred1[0][0]), int(pred2[0][0]), int(pred3[0][0]), int(pred4[0][0]), int(pred5[0][0]))
            # print(int(batch[1][0][0]), int(batch[1][1][0]), int(batch[1][2][0]), int(batch[1][3][0]), int(batch[1][4][0]), int(batch[1][5][0]))
            print(batch[2][0])

            for i in range(len(pred)):
                cm[i][int(batch[1][i][0])][pred[i]] += 1

            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # count += 1
            #
            # if count >= 10:
            #     break

    return cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    opt = parser.parse_args()

    print(opt)

    device_ids = [0]
    device = torch.device("cuda")

    model = Net()
    # model.load_state_dict(torch.load('./model/BCNN_vgg16_epoch_100.pth').state_dict())
    model.load_state_dict(torch.load('./model/BCNN_resnet101_epoch_100.pth').module.state_dict())
    # model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    model.eval()
    summary(model, (3, 224, 224))

    attribute = ["Makename", "Pose", "Car Type", "Color", "Seat Number", "Door Number"]
    cm = batch_test()
    dicts = get_dict()

    for i in range(6):
        print(u"{:-^15}:".format(attribute[i]))
        meanP = 0.0
        countP = 0
        meanR = 0.0
        countR = 0
        for j in range(cm[i].shape[0]):
            TP = cm[i][j][j]
            TP_FP = cm[i][j, :].sum()
            TP_FN = cm[i][:, j].sum()
            if TP_FP != 0:
                meanP += float(TP / TP_FP)
                countP += 1
            if TP_FN != 0:
                meanR += float(TP / TP_FN)
                countR += 1
            print("Class{:^3}:{:^20}, TP:{:^6}, TP+FP:{:^6}, TP+FN:{:^6}, Precision:{:6.4f}, Recall:{:6.4f}".format(j, dicts[i][j], TP, TP_FP, TP_FN, TP / TP_FP, TP / TP_FN))
        meanP /= countP
        meanR /= countR
        print("Average Precision:{:8.4f}, Average Recall:{:8.4f}".format(meanP, meanR))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
