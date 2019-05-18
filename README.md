# MultiLabelClassification
This is a project for Media&amp;Recognition course.

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
This will automatically generate a checkpoint path, and then the .pth.tar file would be written there.

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
    ../dataset
    --gpu=0
    --finetune=1
```
### Restart
Use the parameter '--model' would help you continue your training after a stop. If you want to load the fc layer as well, add the parameter --fc
```Bash
python ./PascalTrain.py\
    ../dataset
    --gpu=0
    --model=../checkpoints/example.pth
    --fc=1
``` 
## Evaluating Guide
Run the Test.py to evaluate your model on a dataset:
```Bash
python ./Test.py\
    {$YOUR_MODEL_PATH}
    --model={$YOUR_MODEL}
    --mode=evaluate
    --testpath={$DATASET_ROOT}
    --gpu=0
```