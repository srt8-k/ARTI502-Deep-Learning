#!/usr/bin/env python3
"""ARTI502 Assignment 2: CIFAR-10 CNN tasks and reproducible external benchmark.

Task 1: load CIFAR-10; Task 2: CNN + training/test functions; Task 3: visualize;
Task 4: augmentations; Task 5: optional CNN training, evaluate external XL model.

External weights attribution:
https://huggingface.co/DeKUT-DSAIL/efficientnetv2-xl-cifar10-upsample
The published CIFAR-10 XL checkpoint was not trained by the authors.
Recorded measured accuracy 99.04% / loss 0.157579 came from original Colab
notebook; running this script again is a new evaluation, not a training run.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(.3), nn.Linear(256, 10))

    def forward(self,x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        loss=F.cross_entropy(logits,y)
        loss.backward()
        optimizer.step()
        n=y.shape[0]
        total_loss += loss.item()*n
        correct += (logits.argmax(1)==y).sum().item()
        total += n
    return total_loss/total,100*correct/total


def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            logits=model(x)
            total_loss += F.cross_entropy(logits.float(),y,reduction='sum').item()
            correct += (logits.argmax(1)==y).sum().item()
            total += y.shape[0]
    return total_loss/total,100*correct/total


def datasets_and_loaders(data_path, batch_size=128):
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    eval_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train_full=datasets.CIFAR10(str(data_path),train=True,download=True,transform=train_tf)
    val_full=datasets.CIFAR10(str(data_path),train=True,download=False,transform=eval_tf)
    indices=torch.randperm(len(train_full),generator=torch.Generator().manual_seed(42)).tolist()
    train=Subset(train_full,indices[:45000]);val=Subset(val_full,indices[45000:])
    test=datasets.CIFAR10(str(data_path),train=False,download=True,transform=eval_tf)
    return (DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=2),
            DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=2),
            test)


def task3_image_grid(data_path, output_dir):
    """Visualize ten genuine CIFAR-10 training images (Task 3)."""
    import matplotlib.pyplot as plt
    dataset = datasets.CIFAR10(str(data_path), train=True,
                               download=False, transform=transforms.ToTensor())
    images = [dataset[i] for i in range(10)]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, (tensor, label) in zip(axes.flat, images):
        ax.imshow(tensor.permute(1, 2, 0))
        ax.set_title(CLASSES[label])
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'Task3_CIFAR10_Visualization.png', dpi=180)
    plt.close(fig)


def augmentation_demo(data_path, output_dir):
    import matplotlib.pyplot as plt
    raw=datasets.CIFAR10(str(data_path),train=True,download=False,transform=transforms.ToTensor())
    tensor,_=raw[0]
    pil=transforms.ToPILImage()(tensor)
    norm=transforms.Normalize(MEAN,STD)(tensor)
    restore=(norm*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]).clamp(0,1)
    gaussian=(tensor+torch.randn_like(tensor)*.1).clamp(0,1)
    salt_pepper=tensor.clone(); mask=torch.rand(1,32,32)
    salt_pepper[:,mask[0]<.025]=0; salt_pepper[:,mask[0]>.975]=1
    examples=[('Original',tensor),('ToPILImage',pil),('Horizontal flip',transforms.RandomHorizontalFlip(p=1)(pil)),
              ('Random resized crop',transforms.RandomResizedCrop(32,scale=(.7,1))(pil)),
              ('Color jitter',transforms.ColorJitter(brightness=.4,contrast=.4,saturation=.4,hue=.1)(pil)),
              ('Rotation',transforms.RandomRotation(25)(pil)),('Normalize (inverse displayed)',restore),
              ('Gaussian noise',gaussian),('Salt and pepper',salt_pepper),('ToTensor',transforms.ToTensor()(pil))]
    fig,ax=plt.subplots(2,5,figsize=(13,6))
    for a,(name,img) in zip(ax.flat,examples):
        if isinstance(img,torch.Tensor): img=img.permute(1,2,0).numpy()
        a.imshow(img);a.set_title(name);a.axis('off')
    fig.tight_layout();fig.savefig(output_dir/'Task4_Augmentation.png',dpi=180)
    plt.close(fig)


def external_xl_test(data_path,weights_path,output_dir,device):
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import shutil
    repo='DeKUT-DSAIL/efficientnetv2-xl-cifar10-upsample'
    if device.type!='cuda':
        raise RuntimeError('The 831MB XL benchmark requires a CUDA GPU in this script.')
    if not weights_path.exists():
        source=hf_hub_download(repo_id=repo,filename='model.safetensors')
        shutil.copy2(source,weights_path)
    model=timm.create_model('tf_efficientnetv2_xl',pretrained=False,num_classes=10)
    model.load_state_dict(load_file(str(weights_path)),strict=True)
    model=model.to(device).eval()
    xl_tf=transforms.Compose([transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
                              transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    test=datasets.CIFAR10(str(data_path),train=False,download=True,transform=xl_tf)
    loader=DataLoader(test,batch_size=8,shuffle=False,num_workers=0)
    confusion=torch.zeros(10,10,dtype=torch.long);loss_sum=0.;correct=total=0
    with torch.inference_mode():
        for x,y in loader:
            x=x.to(device)
            with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
            logits=logits.float();prediction=logits.argmax(1).cpu()
            loss_sum+=F.cross_entropy(logits,y.to(device),reduction='sum').item()
            correct+=(prediction==y).sum().item();total+=len(y)
            confusion+=torch.bincount(y*10+prediction,minlength=100).reshape(10,10)
    result={'model':'EfficientNetV2-XL','external_pretrained_weights':repo,
            'test_accuracy':100*correct/total,'test_loss':loss_sum/total,
            'correct':correct,'total':total,'confusion_matrix':confusion.tolist()}
    (output_dir/'Task5_Detailed_Results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='confusion_matrix'},indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='/content/cifar_data')
    p.add_argument('--save-dir',default='/content/drive/MyDrive/ARTI502_Assignment2')
    p.add_argument('--train-baseline',action='store_true',help='Optional 20-epoch training')
    p.add_argument('--evaluate-xl',action='store_true',help='External pretrained benchmark, CUDA required')
    args=p.parse_args()
    path=Path(args.data_dir);save=Path(args.save_dir);save.mkdir(parents=True,exist_ok=True)
    report=save/'Report_Figures';report.mkdir(exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train,val,test=datasets_and_loaders(path)
    print('Task 1 CIFAR-10:',len(train.dataset),len(val.dataset),len(test),device)
    images,labels=next(iter(train));print('Task 3 tensor sizes:',images.shape,labels.shape)
    task3_image_grid(path,report)
    augmentation_demo(path,report)
    if args.train_baseline:
        model=CNN().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        hist=[]
        for epoch in range(20):
            train_loss,train_acc=train_one_epoch(model,train,optimizer,device)
            val_loss,val_acc=evaluate(model,val,device)
            hist.append({'epoch':epoch+1,'train_loss':train_loss,'train_acc':train_acc,'val_loss':val_loss,'val_acc':val_acc})
            print(hist[-1])
        (report/'optional_baseline_history.json').write_text(json.dumps(hist,indent=2))
        import matplotlib.pyplot as plt
        for metric, ylabel, filename in [
            ('loss', 'Cross-entropy loss', 'Optional_Baseline_Loss.png'),
            ('acc', 'Accuracy (%)', 'Optional_Baseline_Accuracy.png'),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            xx = [row['epoch'] for row in hist]
            ax.plot(xx, [row['train_' + metric] for row in hist], label='Train')
            ax.plot(xx, [row['val_' + metric] for row in hist], label='Validation')
            ax.set(xlabel='Epoch', ylabel=ylabel, title='Optional baseline training: ' + metric)
            ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
            fig.savefig(report / filename, dpi=180)
            plt.close(fig)
    if args.evaluate_xl:
        external_xl_test(path,save/'external_efficientnetv2_xl.safetensors',report,device)
    if not args.train_baseline and not args.evaluate_xl:
        print('No GPU training/evaluation requested. Pass --train-baseline or --evaluate-xl if needed.')

if __name__=='__main__':main()