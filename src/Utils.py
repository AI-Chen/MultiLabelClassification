import os
import numpy as np
import torch
import torch.utils.data as data
from scipy.misc import imread
from PIL import Image
from Network import *
from sklearn.metrics import average_precision_score


models ={
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


def test_map(net, logger, val_loader, steps, gpu, crops):
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

def adjust_learning_rate(optimizer, epoch, init_lr, step=80, decay=0.1):
    """
    This function adjust the learning rate automatically during training.
    https://www.pytorchtutorial.com/pytorch-learning-rate-decay/
    :param optimizer: the optimizer
    :param epoch: current epoch
    :param init_lr: initial learning rate
    :param step: literally
    :param decay: literally
    :return: Nothing
    """
    lr = init_lr*(decay**(epoch//step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_model_from_file(filepath, model="resnet18", load_fc=None):
    """
    Load the trained model from .pth file. Only for the same model trained before
    :param filepath: the path to .pth file
    :param model: the backbone network
    :param load_fc: whether to load fc layer
    :return: loaded model
    """
    # Get the initial network
    dict_init = torch.load(filepath)
    keys = [k for k, v in dict_init.items()]
    keys.sort()
    # Generate a new network
    net = models[model](pretrained=False, num_classes=20)
    model_dict = net.state_dict()
    # load the layers
    to_load = []
    for k in keys:
        if k not in model_dict:
            continue
        if load_fc is not None or 'fc' not in k:
            to_load.append(k)
    # load the dict
    dict_init = {k: v for k, v in dict_init.items() if k in to_load and k in model_dict}
    model_dict.update(dict_init)
    net.load_state_dict(model_dict)

    return net


class Logger:
    def __init__(self, path):
        self.path = path

    def scalar_summary(self, name, value, steps):
        self.__dict__[name] = (steps, value)


class MyDataLoader(data.Dataset):
    def __init__(self, transform, trainval='train', data_path='../dataset', random_crops=0):
        """
        Initialize the dataset. Inherited from torch.data.Dataset, __len__ and __getitem__ need to be implemented.
        VOC(Labels only) tree:
        --dataset root
         |--train
         | |--JPEGImages(dir)
         | |--annotations.txt
         |
         |--test
           |--JPEGImages(dir)
           |--annotations.txt
        :param transform: the transformation
        :param data_path: the root of the datapath
        :param random_crops:
        """
        self.data_path = data_path
        self.transform = transform
        self.random_crops = random_crops
        self.train_or_test = trainval

        self.__init_classes()
        self.names, self.labels = self.__dataset_info()

    def __getitem__(self, index):
        """
        This is the getitem func which enables enumerator. Implemented.
        :param index: the index of the picture
        :return: tuple (picture, its label(s))
        """
        x = imread(os.path.join(self.data_path, self.train_or_test, 'JPEGImages', self.names[index]+'.jpg'),
                   mode='RGB')
        x = Image.fromarray(x)

        # Resize directly instead of the strange operations done below...
        x = x.resize((224, 224), Image.BILINEAR)

        # scale = np.random.rand() * 2 + 0.25
        # w = int(x.size[0] * scale)
        # h = int(x.size[1] * scale)
        # if min(w, h) < 227:
        #     scale = 227 / min(w, h)
        #     w = int(x.size[0] * scale)
        #     h = int(x.size[1] * scale)

        # x = x.resize((w,h), Image.BILINEAR) # Random scale

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.labels[index]
        return x, y

    def __len__(self):
        """
        How many images are there. Implemented.
        :return: length
        """
        return len(self.names)

    def __dataset_info(self):
        """
        Generate names(np.array, with string elements) and labels(np.array, with array(number) elements).
        The labels appears like this: [0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0]
        Those with value 1 means the object exists in this image
        :return: names labels
        """
        annotation_file = os.path.join(self.data_path, self.train_or_test, 'annotations.txt')
        with open(annotation_file, 'r') as fp:
            lines = fp.readlines()

        names = []
        labels = []
        for line in lines:
            # Name
            names.append(line.strip('\n').split(' ')[0])

            # Label
            str_label = line.strip('\n').split(' ')[1:]
            num_label = [int(x) for x in str_label]
            flag_label = np.zeros(self.num_classes)
            flag_label[num_label] = 1

            labels.append(np.array(flag_label))

        return np.array(names), np.array(labels).astype(np.float32)

    def __init_classes(self):
        self.classes = ('person', 'bird', 'cat', 'cow',
                        'dog', 'horse', 'sheep', 'aeroplane', 'bicycle',
                        'boat', 'bus', 'car', 'motorbike',
                        'train', 'bottle', 'chair',
                        'diningtable', 'pottedplant', 'sofa', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))