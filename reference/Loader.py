# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""
import os
import numpy as np
import torch
import torch.utils.data as data
from scipy.misc import imread, imresize
# from scipy.sparse import csr_matrix
from PIL import Image
# import xml.etree.ElementTree as ET


class MyDataLoader(data.Dataset):
    def __init__(self, transform, trainval='train', data_path='../VOC2012', random_crops=0):
        """
        Initialize the dataset.
        VOC(Labels only) tree:
        --root
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
        This is the getitem func which enables the usage of [] operator
        :param index: the index of the picture
        :return: tuple (picture, its label(s))
        """
        x = imread(os.path.join(self.data_path, self.train_or_test, '/JPEGImages/', self.names[index] + '.jpg'),
                   mode='RGB')
        x = Image.fromarray(x)

        # Resize directly instead of the strange operations done below...
        x = x.resize(224, 224)

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
        return len(self.names)

    def __dataset_info(self):
        """
        Generate names(np.array, with string elements) and labels(np.array, with array(number) elements).
        :return: names labels
        """
        annotation_file = os.path.join(self.data_path, 'annotations.txt')
        with open(annotation_file, 'r') as fp:
            lines = fp.readlines()

        names = []
        labels = []
        for line in lines:
            names.append(line.split('\0')[0])
            labels.append(np.array(line.split('\0')[1:]))

        return np.array(names), np.array(labels).astype(np.float32)


        # annotation_files = os.listdir(self.data_path+'/Annotations')
        # with open(self.data_path + '/ImageSets/Main/' + self.trainval + '.txt') as f:
        #     annotations = f.readlines()
        #
        # annotations = [n[:-1] for n in annotations]
        #
        # names = []
        # labels = []
        # for af in annotations:
        #     if len(af) != 6:
        #         continue
        #     filename = os.path.join(self.data_path, 'Annotations', af)
        #     tree = ET.parse(filename + '.xml')
        #     objs = tree.findall('object')
        #     num_objs = len(objs)
        #
        #     boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        #     boxes_cl = np.zeros((num_objs), dtype=np.int32)
        #
        #     for ix, obj in enumerate(objs):
        #         bbox = obj.find('bndbox')
        #         # Make pixel indexes 0-based
        #         x1 = float(bbox.find('xmin').text) - 1
        #         y1 = float(bbox.find('ymin').text) - 1
        #         x2 = float(bbox.find('xmax').text) - 1
        #         y2 = float(bbox.find('ymax').text) - 1
        #
        #         cls = self.class_to_ind[obj.find('name').text.lower().strip()]
        #         boxes[ix, :] = [x1, y1, x2, y2]
        #         boxes_cl[ix] = cls
        #
        #     lbl = np.zeros(self.num_classes)
        #     lbl[boxes_cl] = 1
        #     labels.append(lbl)
        #     names.append(af)
        #
        # return np.array(names), np.array(labels).astype(np.float32)


    def __init_classes(self):
        self.classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))