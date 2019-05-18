import torch
import torchvision.transforms as transforms
import argparse

from Utils import predict, eval_macc, MyDataLoader,eval_wacc


parser = argparse.ArgumentParser(description='Predict a picture or evaluate the model on a test dataset')
parser.add_argument("modelpath", type=str, help="The model for prediction or evaluation")
parser.add_argument("--mode", type=str, default="evaluate", choices=["predict", "evaluate", "evalmacc", "evalwacc"],
                    help="Whether to predict a single image or evaluate a model on a dataset")
parser.add_argument("--testpath", type=str, required=True, help="The path to the test image or dataset")
parser.add_argument("--gpu", type=int, default=None, help="Which gpu to use(leave it None for cpu)")
parser.add_argument("--model", type=str, required=True, help="Which kind of the model is the one for test")
parser.add_argument("--crops", type=int, default=0, help="How many crops while testing")
parser.add_argument("--batch", type=int, default=8, help="Batch size while evaluating")
args = parser.parse_args()

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if args.mode == "predict":
        predict(val_transform, model_path=args.modelpath, img_path=args.testpath, model=args.model, gpu=args.gpu)
    else:
        val_data = MyDataLoader(transform=val_transform, trainval='test', data_path=args.testpath,
                                random_crops=args.crops)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch, shuffle=False, num_workers=4)
        if args.mode == "evalmacc":
            eval_macc(val_loader, model_path=args.modelpath, model=args.model, gpu=args.gpu, crops=args.crops)
        if args.mode == "evalwacc":
            eval_wacc(val_loader, model_path=args.modelpath, model=args.model, gpu=args.gpu, crops=args.crops)
        if args.mode == "evaluate":
            eval_macc(val_loader, model_path=args.modelpath, model=args.model, gpu=args.gpu, crops=args.crops)
            eval_wacc(val_loader, model_path=args.modelpath, model=args.model, gpu=args.gpu, crops=args.crops)
