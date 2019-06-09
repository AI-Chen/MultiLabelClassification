# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import argparse

# sys.path.append('../Utils')
# TODO: Finish the class Logger(I don't actually know what it is used for)

import torch
import torchvision.transforms as transforms

CORES = 4  # int(float(multiprocessing.cpu_count())*0.25)

from Network import resnet18, resnet34, resnet50, resnet101, resnet152
from Utils import MyDataLoader, adjust_learning_rate, load_model_from_file, compute_mAP, eval_map

parser = argparse.ArgumentParser(description='Train network on Pascal VOC 2012')
parser.add_argument('pascal_path', type=str, help='Path to Pascal VOC 2012 folder')
parser.add_argument('--finetune', default=None, type=int, help='whether to use pytorch pretrained model and finetune')
parser.add_argument('--model', default='resnet18', type=str, help='which backbone network to use',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--modelpath', default=None, type=str, help='pretrained model path')
parser.add_argument('--fc', default=None, type=int, help='load fc6 and fc7 from model')
parser.add_argument('--gpu', default=None, type=int, help='gpu id')
parser.add_argument('--epochs', default=160, type=int, help='max training epochs')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--checkpoint', default='../checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--crops', default=10, type=int, help='number of random crops during testing')
args = parser.parse_args()
# args = parser.parse_args([
#    '../dataset',
#    '--gpu','0',
#    '--finetune','1'
#    '--model','resnet18'
# ])
prefix = time.strftime("%y%m%d_%H%M", time.localtime())
models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}


def main():
    # Training devices
    if args.gpu is not None:
        print('Using GPU %d' % args.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print('CPU mode')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    val_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    # Load the dataset. When training, disable the random crop.
    train_data = MyDataLoader(transform=train_transform, trainval='train', data_path=args.pascal_path, random_crops=0)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch, 
                                               shuffle=True,
                                               num_workers=CORES)

    val_data = MyDataLoader(transform=val_transform, trainval='test', data_path=args.pascal_path,
                            random_crops=args.crops)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=args.batch, 
                                             shuffle=False,
                                             num_workers=CORES)
    
    N = len(train_data.names)
    iter_per_epoch = int(N/args.batch)

    # Network initialize
    # finetune: freeze some layers and modify the fc layer.
    if args.finetune is not None:
        # Initialize the network
        net = models[args.model](pretrained=True)
        # Freeze conv layers
        for i, (name, param) in enumerate(net.named_parameters()):
            if 'conv' in name:
                param.requires_grad = False
        # Modify the fc layer
        in_channel = net.fc.in_features
        net.fc = torch.nn.Linear(in_features=in_channel, out_features=20)

    elif args.modelpath is not None:
        net = load_model_from_file(args.modelpath, model=args.model, load_fc=args.fc)

    else:
        net = models[args.model](pretrained=False, num_classes=20)

    if args.gpu is not None:
        net.cuda()

    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=args.lr, momentum=0.9, weight_decay=0.0001)
    

    ############## TRAINING ###############
    print("Start training, lr: %f, batch-size: %d" % (args.lr, args.batch))
    print("Model: " + args.model)
    print("Checkpoint Path: "+args.checkpoint)
    print("Time: "+prefix)
    if args.modelpath is not None:
        print("Training from past model: "+args.modelpath)
    
    # Train the Model
    steps = args.iter_start
    for epoch in range(iter_per_epoch*args.iter_start, args.epochs):
        adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
        
        mAP = []
        for i, (images, labels) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(images)
            
            mAP.append(compute_mAP(labels.data, outputs.data))
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = loss.cpu().data.numpy()
            
            if steps % 100 == 0:
                print('[%d/%d] %d), Loss: %.3f, mAP %.2f%%' % (epoch+1, args.epochs, steps, loss,100*np.mean(mAP[-20:])))
            
            steps += 1
        
        if epoch % 5 == 0:
            filename = '%s/%s_%s_%03i.pth' % (args.checkpoint, args.model, prefix, epoch+1)
            torch.save(net.state_dict(), filename)
            print('Saved: '+args.checkpoint+"/"+filename)

            eval_map(net, None, val_loader, steps, args.gpu, args.crops)
        
        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break


if __name__ == "__main__":
    main()