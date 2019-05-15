import torch
import numpy as np
import os
from scipy.misc import imread
from PIL import Image
import torchvision.transforms as transforms

from Utils import load_model_from_file


def test(transform, model_path='../checkpoints/resnet18_190515_1825_001.pth', img_path='../test.jpg', model="resnet18", gpu=None):
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
    print(outputs)


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    test(val_transform)
