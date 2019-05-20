# MultiLabelClassification
This is a project for Media&amp;Recognition course.

Reference: https://github.com/bbrattoli/PascalClassificationPytorch

## Preprocess
For training on VOC2012 with 09-12 pictures, and test on 07-08(our homework requirement actually), run:
```bash
python ./gen_dataset.py {$YOUR_DATASET_PATH}
```
Note: All the required dataset path is the root to the dataset path with required strcture.
Namely the structure should be like this:

    --dataset_root
     |--origin
     | |--JPEGImages(dir)
     | |--annotations.txt
     |
     |--train
     | |--JPEGImages(dir)
     | |--annotations.txt
     |
     |--test
       |--JPEGImages(dir)
       |--annotations.txt

And YOUR_DATASET_PATH is the dataset_root above.

'origin' is the original dataset, which is PascalVOC2012 in our homework. It differs from standard VOC2012
dataset in that it only has one annotation file in it, with the format: $FILENAME(without suffix) label0 label1... each line.

If you want to train it on other datasets, perhaps you should do the preprocess yourself, and upon
training you just need to make sure you have correct file structure like above(origin is not needed)

## Training Guide

### Training from scratch
The working directory is src/, first enter the directory:
```Bash
cd src
```
Then run:
```Bash
python ./PascalTrain.py\
    {$YOUR_DATASET_PATH}\
    --gpu={$YOUR_GPU_INDEX}
```
This will automatically generate a checkpoint path, and then the .pth checkpoint file would be written there.

You can use --epochs, --batch, --lr to appoint the parameters too:
```Bash
python ./PascalTrain.py\
    {$YOUR_DATASET_PATH}\
    --gpu={$YOUR_GPU_INDEX}\
    --epochs={$MAX_TRAINING_EPOCH}\
    --batch={$BATCH_SIZE}\
    --lr={$LEARNING_RATE}
```

### Finetuning(May need Internet)
Use the parameter '--finetune' would enable the usage of pytorch implemented pretrained model. 
Then the layers close to input would be frozen, and the fc layer will be modified for finetuning the model pretrained on ImageNet(1000 classes)<br>
For example:
```Bash
python ./PascalTrain.py\
    ../dataset\
    --gpu=0\
    --finetune=1
```
### Restart
Use the parameter '--model' would help you continue your training after a stop. If you want to load the fc layer as well, add the parameter --fc
```Bash
python ./PascalTrain.py\
    ../dataset\
    --gpu=0\
    --model=../checkpoints/example.pth\
    --fc=1
``` 
## Evaluating Guide
Run the Test.py to evaluate your model on a dataset(output: mAcc wAcc):
```Bash
python ./Test.py\
    {$YOUR_MODEL_PATH}\
    --model={$YOUR_MODEL}\
    --mode=evaluate\
    --testpath={$DATASET_ROOT}\
    --gpu=0
```
mAP is not a criterion of our homework, so I did not add it to Test.py. However, there is a eval_map function in Utils.py
which has the same input and output with eval_macc. So you can add it manually if needed.

If you want to predict the labels of a single image, run:
```bash
python ./Test.py\
    {$YOUR_MODEL_PATH}\
    --model={$YOUR_MODEL}\
    --mode=predict\
    --testpath={$IMAGE_PATH}\
    --gpu=0
```