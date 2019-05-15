import torch
import numpy as np
import os
from scipy.misc import imread
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

from Utils import load_model_from_file, MyDataLoader


def gethreshold(val_loader, transform, gpu=None, model_path='../checkpoints/resnet18_190515_1825_001.pth', model="resnet18"):
    net = load_model_from_file(model_path, model, True)
    if gpu is not None:
        net.cuda()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    net.eval()
    ymin = np.zeros(20)
    ymax = np.linspace(-100, -100, 20)
    for i, (images, labels) in enumerate(val_loader):
        images = images.view((-1, 3, 224, 224))
        if gpu is not None:
            images = images.cuda()

        outputs = net(images)
        outputs = outputs.cpu().data
        outputs = outputs.view((-1, 20))
        y_true = labels.cpu().numpy()
        y_pred = outputs.cpu().numpy()
        for k in range(0, val_loader.batch_size):
            for j in range(20):
                if y_true[k][j] == 0:
                    if y_pred[k][j] > ymax[j]:
                        ymax[j] = y_pred[k][j]
                else:
                    if y_pred[k][j] < ymin[j]:
                        ymin[j] = y_pred[k][j]

        if i % 20 == 0:
            print('ymin:', ymin)
            print('ymax:', ymax)
    return ymin, ymax


def test(transform, model_path='../checkpoints/190512_1711_011.pth', img_path='../test.jpg', model="resnet18",
         gpu=None):
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
    with torch.no_grad():
        img = Variable(img)
    if gpu is not None:
        img = img.cuda()

    outputs = net(img)
    outputs = outputs.cpu().data
    outputs = outputs.view((-1, 20))
    print(outputs)


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    val_data = MyDataLoader(transform=val_transform, trainval='test', data_path='../dataset/',
                            random_crops=10)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=8,
                                             shuffle=False,
                                             num_workers=4)

    # test(val_transform, gpu=0)
    gethreshold(val_loader, val_transform, gpu=0)
