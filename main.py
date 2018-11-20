from __future__ import print_function
import os
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary

# from BCNN_VGG16 import Net
from BCNN_ResNet101 import Net
from data import get_training_set, get_val_set, get_test_set

def l1_penalty(var):
    return torch.abs(var).sum()

def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())

def train(epoch):
    model.train()

    for batch_idx, batch in enumerate(training_data_loader, 1):
        data = batch[0].to(device)
        target0 = batch[1][0]
        target1 = batch[1][1]
        target2 = batch[1][2]
        target3 = batch[1][3]
        target4 = batch[1][4]
        target5 = batch[1][5]

        target0 = torch.tensor(target0).to(device)
        target1 = torch.tensor(target1).to(device)
        target2 = torch.tensor(target2).to(device)
        target3 = torch.tensor(target3).to(device)
        target4 = torch.tensor(target4).to(device)
        target5 = torch.tensor(target5).to(device)

        optimizer.zero_grad()
        output = model(data)
        loss0 = criterion(output[0], target0) + opt.lambda2 * l2_penalty(output[0])
        loss1 = criterion(output[1], target1) + opt.lambda2 * l2_penalty(output[1])
        loss2 = criterion(output[2], target2) + opt.lambda2 * l2_penalty(output[2])
        loss3 = criterion(output[3], target3) + opt.lambda2 * l2_penalty(output[3])
        loss4 = criterion(output[4], target4) + opt.lambda2 * l2_penalty(output[4])
        loss5 = criterion(output[5], target5) + opt.lambda2 * l2_penalty(output[5])
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5

        # loss.register_hook(lambda g : print(g))
        # loss0.register_hook(lambda g : print(g))
        # loss1.register_hook(lambda g : print(g))
        # loss2.register_hook(lambda g : print(g))
        # loss3.register_hook(lambda g : print(g))
        # loss4.register_hook(lambda g : print(g))
        # loss5.register_hook(lambda g : print(g))

        loss.backward()
        optimizer.step()
        # optimizer.module.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                epoch, batch_idx * len(data), len(training_data_loader.dataset),
                       100. * batch_idx / len(training_data_loader), loss0.item(), loss1.item(), loss2.item(),
                loss3.item(), loss4.item(), loss5.item(), ))
            # for param_lr in optimizer.module.param_groups:
            for param_lr in optimizer.param_groups:
                print('lr_rate: ' + str(param_lr['lr']))

def val():
    model.eval()

    val_loss = 0
    correct0 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0
    with torch.no_grad():
        for batch in validating_data_loader:
            input = batch[0].to(device)
            target0 = batch[1][0]
            target1 = batch[1][1]
            target2 = batch[1][2]
            target3 = batch[1][3]
            target4 = batch[1][4]
            target5 = batch[1][5]

            target0 = torch.tensor(target0).to(device)
            target1 = torch.tensor(target1).to(device)
            target2 = torch.tensor(target2).to(device)
            target3 = torch.tensor(target3).to(device)
            target4 = torch.tensor(target4).to(device)
            target5 = torch.tensor(target5).to(device)

            prediction = model(input)
            val_loss += criterion(prediction[0], target0).item()
            val_loss += criterion(prediction[1], target1).item()
            val_loss += criterion(prediction[2], target2).item()
            val_loss += criterion(prediction[3], target3).item()
            val_loss += criterion(prediction[4], target4).item()
            val_loss += criterion(prediction[5], target5).item()

            pred0 = prediction[0].max(1, keepdim=True)[1]
            pred1 = prediction[1].max(1, keepdim=True)[1]
            pred2 = prediction[2].max(1, keepdim=True)[1]
            pred3 = prediction[3].max(1, keepdim=True)[1]
            pred4 = prediction[4].max(1, keepdim=True)[1]
            pred5 = prediction[5].max(1, keepdim=True)[1]

            correct0 += pred0.eq(target0.view_as(pred0)).sum().item()
            correct1 += pred1.eq(target1.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target2.view_as(pred2)).sum().item()
            correct3 += pred3.eq(target3.view_as(pred3)).sum().item()
            correct4 += pred4.eq(target4.view_as(pred4)).sum().item()
            correct5 += pred5.eq(target5.view_as(pred5)).sum().item()

    val_loss /= len(validating_data_loader.dataset)
    print(
        '\nValidate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%) ({:.0f}%)\n'.format(
            val_loss, correct0, len(validating_data_loader.dataset),
            100. * correct0 / len(validating_data_loader.dataset),
            100. * correct1 / len(validating_data_loader.dataset),
            100. * correct2 / len(validating_data_loader.dataset),
            100. * correct3 / len(validating_data_loader.dataset),
            100. * correct4 / len(validating_data_loader.dataset),
            100. * correct5 / len(validating_data_loader.dataset),
        ))

def checkpoint(epoch):
    model_out_path = "./model/BCNN_resnet101_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--valBatchSize', type=int, default=10, help='validating batch size')
    parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 of Adam. Default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 of Adam. Default=0.999')
    parser.add_argument('--lambda1', type=float, default=0.5, help='Lambda1 of L1. Default=0.5')
    parser.add_argument('--lambda2', type=float, default=0.01, help='Lambda2 of Lambda2. Default=0.01')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum. Default=0.9')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    opt = parser.parse_args()
    print(opt)

    device_ids = [0, 1]
    device = torch.device("cuda")

    torch.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = get_training_set()
    val_set = get_val_set()
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    validating_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.valBatchSize, shuffle=False)

    print('===> Building model')

    model = Net()

    if opt.snapshot is not None:
        print('Loading model from {}...'.format(opt.snapshot))
        model.load_state_dict(torch.load(opt.snapshot).module.state_dict())

    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    summary(model, (3, 224, 224))

    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = optim.SGD(model.parameters(), momentum=opt.momentum, lr=opt.lr)
    optimizer = optim.Adam(model.parameters(), betas=(opt.beta1, opt.beta2), lr=opt.lr)
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        val()
        if(epoch % 50 == 0):
           checkpoint(epoch)
           for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5