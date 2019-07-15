“”“
This is added to the project for manually figure out the best threshold in predicting.
The file is not directly depended by any of this project, so I put it outside along with preprocess
”“”
import torch
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

from Utils import predict, eval_macc, MyDataLoader, eval_wacc, eval_map, eval_f1, load_model_from_file

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    testpath = "../dataset"
    batch = 10
    modelpath = "../checkpoints/resnet18_190515_2049_001.pth"
    gpu = "0"
    val_data = MyDataLoader(transform=val_transform, trainval='test', data_path=testpath,
                            random_crops=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch, shuffle=False, num_workers=4)
    net = load_model_from_file(modelpath, "resnet18", True)
    if gpu is not None:
        net.cuda()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    pos = []
    neg = []
    net.eval()
    for idx, (images, labels) in enumerate(val_loader):
        images = images.view((-1, 3, 224, 224))
        if gpu is not None:
            images = images.cuda()

        outputs = net(images)
        outputs = outputs.cpu().data
        outputs = outputs.view((-1, 20))

        # outputs: shape [batchsize * num_classes]
        for i, lbl in enumerate(labels):
            # print(lbl, outputs)
            if lbl[0] == 1:
                pos.append(outputs[i].numpy())
            else:
                neg.append(outputs[i].numpy())

        print("Evaluating threshold, Batch_size: %d" % idx, end='\r')

    pos = np.array(pos)
    neg = np.array(neg)
    plt.hist(pos[:, 0])
    plt.savefig("figure1.jpg")
    plt.close()
    plt.hist(neg[:, 0])
    plt.savefig("figure2.jpg")
    plt.close()
