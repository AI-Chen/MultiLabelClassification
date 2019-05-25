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
    Evaluate a model on a dataset, using mAcc as index
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
            outputs = outputs.max(dim=1)[0].view((-1, 20))
        else:
            outputs = outputs.view((-1, 20))

        # outputs: shape [batchsize * num_classes]
        outputs = (outputs > 0)
        acc.append(np.sum((outputs.numpy() == labels.numpy()).astype(float)) / (val_loader.batch_size * 20))

        print("Evaluating mAcc, Batch_size: %d" % idx, end='\r')

    macc = sum(acc) / len(acc)
    print("\nFinal mAcc: %f" % macc)
    return macc


def eval_wacc(val_loader, model_path="../checkpoints/resnet18_190515_2049_001.pth",
              model="resnet18", gpu=None, crops=0):
    """
    Evaluate a model on a dataset, using wAcc as index
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

    acc = np.zeros(20)
    net.eval()
    freq = np.zeros(20)

    for idx, (images, labels) in enumerate(val_loader):
        # Frequency of the labels
        freq += np.sum(labels.numpy(), axis=0)
        images = images.view((-1, 3, 224, 224))

        if gpu is not None:
            images = images.cuda()
        outputs = net(images)
        outputs = outputs.cpu().data
        if crops != 0:
            outputs = outputs.view((-1, crops, 20))
            outputs = outputs.max(dim=1)[0].view((-1, 20))
        else:
            outputs = outputs.view((-1, 20))
        outputs = (outputs > 0)
        acc += np.sum((outputs.numpy() == labels.numpy()), axis=0).astype(float)

        print("Evaluating wAcc, Batch_size: %d" % idx, end="\r")

    freq = freq / np.sum(freq)
    acc = acc / len(val_loader.dataset)

    wacc = np.dot(freq, acc)
    print("\nFinal wAcc: %f" % wacc)
    return wacc


def predict(transform, model_path='../checkpoints/190513.2359_011_0.917.pth', img_path='../test.jpg', model="resnet18",
            crops=0, gpu=None):
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
    if crops == 0:
        img = transform(img)
    else:
        img_crop = []
        for i in range(0, crops):
            img_crop.append(transform(img))
        img = torch.stack(img_crop)
    img = img.view((-1, 3, 224, 224))
    if gpu is not None:
        img = img.cuda()

    outputs = net(img)
    outputs = outputs.cpu().data
    if crops != 0:
        outputs = outputs.view((-1, crops, 20))
        outputs = outputs.max(dim=1)[0].view((-1, 20))
    else:
        outputs = outputs.view((-1, 20))
    print("output tensor:", outputs)
    print("Results:", (outputs > 0) * 1)
    Categories = np.array(['person', 'bird', 'cat', 'cow',
                           'dog', 'horse', 'sheep', 'aeroplane', 'bicycle',
                           'boat', 'bus', 'car', 'motorbike',
                           'train', 'bottle', 'chair',
                           'diningtable', 'pottedplant', 'sofa', 'tvmonitor'])
    print("Categories:", Categories[np.where(outputs[0].numpy() > 0)])


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
    lr = init_lr * (decay ** (epoch // step))
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
        x = imread(os.path.join(self.data_path, self.train_or_test, 'JPEGImages', self.names[index] + '.jpg'),
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
