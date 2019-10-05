import cv2
import warnings
warnings.simplefilter("ignore")
import torch.optim
import torch.nn as nn
import torch backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from dataset import *
import numpy as np
import time
import argparse
import datetime
import os, sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='ce7454')
parser.add_argument('--model', type=str, default="restnet", help='model name')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size - default:128')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs - default:10')
parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate - default:0.005')
parser.add_argument('--lr_decay', type=int, default=20, help='decay lr by 10 after _ epochs - default:20')
parser.add_argument('--input_size', type=int, default=96, help='input size of the depth image - default:96')
parser.add_argument('--augment_probability', type=float, default=1.0, help='augment probability - default:1.0')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum - default:0.9')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay - default:0.0005')
parser.add_argument('--checkpoint', type=str, default=None, help='path/to/checkpoint.pth.tar - default:None')
parser.add_argument('--print_interval', type=int, default=500, help='print interval - default:500')
parser.add_argument('--save_dir', type=str, default="experiments/", help='path/to/save_dir - default:experiments/')
parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models. If none, it will be saved as the date and time')
parser.add_argument('--finetune', action='store_true', help='use a pretrained checkpoint - default:false')

def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    # save to the disk
    expr_dir = os.path.join(opt.save_dir, opt.name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def adjust_learning_rate(optimizer, epoch, args):
    """ Sets the learning rate to the initial LR decayed by 10 every args.lr_decay epochs"""
    # lr = 0.00005
    lr = args.lr * (0.1 ** (epoch // args.lr_decay))
    # print("LR is " + str(lr) + " at epoch " + str(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def set_default_args(args):
    if not args.name:
        now = datetime.datetime.now()
        args.name = now.strftime("%Y-%m-%d-%H-%M")
    args.is_cuda = torch.cuda.is_available()
    args.expr_dir = os.path.join(args.save_dir, args.name)

def main(args):
    set_default_args(args)

    # ADD YOUR MODEL NAME HERE
    if args.model == 'resnet':
        model = models.resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 14)

    model.float()
    if args.is_cuda: model.cuda()
    model.apply(weight_init)
    cudnn.benchmark = True
    criterion = nn.BCEWithLogitsLoss()

    xforms_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224)])
    xforms_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

    train_dataset = CheXpertDataset(args, training=True, transforms=xforms_train)
    valid_dataset = CheXpertDataset(args, training=False, transforms=xforms_val)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=False)

    current_epoch = 0

    if args.checkpoint:
        model, optimizer, current_epoch = load_checkpoint(args.checkpoint, model, optimizer)
        if args.finetune:
            current_epoch = 0

    best = False

    print_options(args)
    training_loss, val_loss, time_taken = ([] for i in range(3))
    best = False
    thres = 0

    for epoch in range(current_epoch, args.epoch):

        optimizer = adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        epoch_train_loss, TT = train(train_loader, model, criterion, optimizer, epoch, args)
        training_loss = training_loss + [epoch_train_loss]
        # evaluate on validation set
        loss_val = validate(val_loader, model, criterion, args)
        val_loss = val_loss + [loss_va]

        state = {
            'epoch': epoch,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer:state_dict(),
        }

        if not os.path.isfile(os.path.join(args.expr_dir, 'model_best.pth.tar')):
            save_checkpoint(state, True, args)

        if (epoch > 1):
            best = (loss_val < min(val_loss[:len(val_loss)-1]))
            if best:
                print("saving best performing checkpoint on val")
                save_checkpoint(state, True, args)

        save_checkpoint(state, False, args)

    save_plt([training_loss, val_loss], ["train_loss", "val_loss"], args)
    save_plt([time_taken], ["time_taken"], args)