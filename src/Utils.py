import os
import numpy as np
import torch
import torch.utils.data as data
from scipy.misc import imread
from PIL import Image
from Network import *
from sklearn.metrics import average_precision_score

models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)


def eval_map(net, logger, val_loader, steps, gpu, crops):
    mAP = []
    net.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = images.view((-1, 3, 224, 224))
        if gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data
        if crops != 0:
            outputs = outputs.view((-1, crops, 20))
            outputs = outputs.mean(dim=1).view((-1, 20))
        else:
            outputs = outputs.view((-1, 20))

        # score = tnt.meter.mAPMeter(outputs, labels)
        mAP.append(compute_mAP(labels, outputs))

    if logger is not None:
        logger.scalar_summary('mAP', np.mean(mAP), steps)
    print('TESTING: %d), mAP %.2f%%' % (steps, 100 * np.mean(mAP)))
    net.train()


def eval_macc(val_loader, model_path="../checkpoints/resnet18_190515_2049_001.pth",
              model="resnet18", gpu=None, crops=0):
    """
    Evaluate a model on a dataset.
    :param val_loader: the dataloader(torch.utils.dataloader) object
    :param model_path: the path to the model
    :param model: which kind is the model
    :param gpu: which gpu to use
    :param crops: how many random crops
    :return: mAcc on the dataset
    """
    net = load_model_from_file(model_path, model=model, load_fc=True)
    if gpu is not None:
        net.cuda()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    acc = []
    net.eval()
    for idx, (images, labels) in enumerate(val_loader):
        images = images.view((-1, 3, 224, 224))
        if gpu is not None:
            images = images.cuda()

        outputs = net(images)
        outputs = outputs.cpu().data
        if crops != 0:
            outputs = outputs.view((-1, crops, 20))
            outputs = outputs.mean(dim=1).view((-1, 20))
        else:
            outputs = outputs.view((-1, 20))

        # outputs: shape [batchsize * num_classes]
        outputs = (outputs > 0)
        acc.append(np.sum((outputs.numpy() == labels.numpy()).astype(float))/(val_loader.batch_size*20))

        if idx % 20 == 0:
            macc = sum(acc) / len(acc)
            print("Batch %d, mAcc: %f " % (idx, macc))

    macc = sum(acc) / len(acc)
    print("Final mAcc: %f", macc)
    return macc


def predict(transform, model_path='../checkpoints/190513.2359_011_0.917.pth', img_path='../test.jpg', model="resnet18",
            gpu=None):
    """
    Predict a image with the model
    :param transform: the torchvision.transforms object. Proper transform may help prediction
    :param model_path: the path to the model
    :param img_path: the path to the image
    :param model: the species of the model
    :param gpu: which gpu to use
    :return: None. The result will be show on the screen directly
    """
    net = load_model_from_file(model_path, model, True)
    if gpu is not None:
        net.cuda()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    net.eval()
    img = imread(img_path, mode='RGB')
    img = Image.fromarray(img)
    img = transform(img)
    img = img.view((-1, 3, 224, 224))
    if gpu is not None:
        img = img.cuda()

    outputs = net(img)
    outputs = outputs.cpu().data
    outputs = outputs.view((-1, 20))
    print("output tensor:", outputs)
    print("Results:", (outputs > 0) * 1)