import os
import shutil
import argparse


def gen_dataset(dataset_root="/home/tsinghuaee13/dataset"):
    origin_path = os.path.join(dataset_root, "origin")
    train_path = os.path.join(dataset_root, "train")
    test_path = os.path.join(dataset_root, "test")

    if not os.path.exists(train_path):
        os.mkdir(train_path)
        os.mkdir(os.path.join(train_path, "JPEGImages"))

    if not os.path.exists(test_path):
        os.mkdir(test_path)
        os.mkdir(os.path.join(test_path, "JPEGImages"))
    # copy images
    for (idx, img_name) in enumerate(os.listdir(os.path.join(origin_path, "JPEGImages"))):
        if img_name[0:4] == "2007" or img_name[0:4] == "2008":
            shutil.copy(os.path.join(origin_path, "JPEGImages", img_name),
                        os.path.join(test_path, "JPEGImages", img_name))
            print("image written, index%d" % idx, end='\r')
        else:
            shutil.copy(os.path.join(origin_path, "JPEGImages", img_name),
                        os.path.join(train_path, "JPEGImages", img_name))
            print("image written, index%d" % idx, end='\r')

    with open(os.path.join(origin_path, "annotations.txt"), "r") as fin:
        with open(os.path.join(test_path, "annotations.txt"), "w") as fout:
            for line in fin.readlines():
                if line[0:4] == "2007" or line[0:4] == "2008":
                    fout.write(line)
        with open(os.path.join(train_path, "annotations.txt"), "w") as fout:
            for line in fin.readlines():
                if line[0:4] == "2009" or line[0:4] == "2010" or line[0:4] == "2011" or line[0:4] == "2012":
                    fout.write(line)


parser = argparse.ArgumentParser(description="A tool used to generate train set and test set")
parser.add_argument("datasetpath", default="/home/tsinghuaee13/dataset")
args = parser.parse_args()

if __name__ == "__main__":
    gen_dataset(args.datasetpath)
