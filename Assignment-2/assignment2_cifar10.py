"""ARTI 502 Assignment 2 - CIFAR-10 CNN experiments.
Reproducible TRAIN/VAL/TEST implementation; GPU needed for full training.
Pretrained EfficientNetV2-M fine-tuning is in efficientnetv2m.py.
"""
import argparse, json, random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

MEAN=(.4914,.4822,.4465)
STD=(.2470,.2435,.2616)

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(128*4*4,256),
                                      nn.ReLU(),nn.Dropout(.3),nn.Linear(256,10))
    def forward(self,x):
        return self.classifier(self.features(x))

class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        c=3
        for c1,c2 in [(64,64),(128,128),(256,256)]:
            for outc in (c1,c2):
                layers += [nn.Conv2d(c,outc,3,padding=1,bias=False),
                           nn.BatchNorm2d(outc),nn.ReLU()]
                c=outc
            layers += [nn.MaxPool2d(2)]
        self.features=nn.Sequential(*layers)
        self.classifier=nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(),nn.Dropout(.3),nn.Linear(256,10))
    def forward(self,x):
        return self.classifier(self.features(x))

def resnet18_cifar():
    model=resnet18(weights=None)
    model.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity()
    model.fc=nn.Linear(model.fc.in_features,10)
    return model

def get_loaders(root,batch,seed=42):
    train_tf=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.ColorJitter(brightness=.1,contrast=.1,saturation=.1),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    eval_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    a=datasets.CIFAR10(root=root,train=True,download=True,transform=train_tf)
    b=datasets.CIFAR10(root=root,train=True,download=False,transform=eval_tf)
    test=datasets.CIFAR10(root=root,train=False,download=True,transform=eval_tf)
    ids=torch.randperm(50000,generator=torch.Generator().manual_seed(seed)).tolist()
    kwargs=dict(batch_size=batch,num_workers=2,pin_memory=torch.cuda.is_available())
    return (DataLoader(Subset(a,ids[:45000]),shuffle=True,**kwargs),
            DataLoader(Subset(b,ids[45000:]),shuffle=False,**kwargs),
            DataLoader(test,shuffle=False,**kwargs),a)

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    loss_sum=correct=total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out=model(x)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        n=y.size(0); total+=n; loss_sum+=loss.item()*n
        correct+=(out.argmax(1)==y).sum().item()
    return loss_sum/total,100*correct/total

@torch.inference_mode()
def evaluate(model,loader,criterion,device,flip=False):
    model.eval()
    loss_sum=correct=total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        logits=model(x)
        if flip:
            probs=(logits.softmax(1)+model(x.flip(-1)).softmax(1))/2
            loss=nn.functional.nll_loss(probs.clamp_min(1e-12).log(),y)
            pred=probs.argmax(1)
        else:
            loss=criterion(logits,y)
            pred=logits.argmax(1)
        n=y.size(0);total+=n;loss_sum+=loss.item()*n
        correct+=(pred==y).sum().item()
    return loss_sum/total,100*correct/total

def show_data(dataset,outpath):
    import matplotlib.pyplot as plt
    classes=dataset.classes
    fig,axes=plt.subplots(2,5,figsize=(12,5))
    mean=torch.tensor(MEAN)[:,None,None]
    std=torch.tensor(STD)[:,None,None]
    for i,ax in enumerate(axes.flat):
        im,lab=dataset[i]
        ax.imshow((im*std+mean).clamp(0,1).permute(1,2,0))
        ax.set_title(classes[lab])
        ax.axis('off')
    fig.tight_layout();fig.savefig(outpath,dpi=180);plt.close(fig)

def plot_history(h,outpath):
    epochs=range(1,len(h['train_loss'])+1)
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(epochs,h['train_loss']);ax[0].set_title('Train loss')
    ax[1].plot(epochs,h['val_loss']);ax[1].set_title('Validation loss')
    ax[2].plot(epochs,h['val_acc']);ax[2].set_title('Validation accuracy')
    for a in ax:
        a.set_xlabel('Epoch');a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(outpath,dpi=180);plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['baseline','deeper','resnet18'],default='baseline')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--data-root',default='./cifar_data')
    parser.add_argument('--out',default='./results')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train,val,test,original=get_loaders(args.data_root,args.batch_size,args.seed)
    print('Images batch shape:',next(iter(train))[0].shape)
    print('Data partitions:',len(train.dataset),len(val.dataset),len(test.dataset))
    show_data(original,outdir/'cifar10_samples.png')
    model={'baseline':BaselineCNN,'deeper':DeeperCNN,'resnet18':resnet18_cifar}[args.model]().to(device)
    criterion=nn.CrossEntropyLoss()
    if args.model=='resnet18':
        optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,
                                  weight_decay=.0005,nesterov=True)
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    history={k:[] for k in ['train_loss','train_acc','val_loss','val_acc']}
    best=-1.;best_epoch=0
    for epoch in range(args.epochs):
        tl,ta=train_one_epoch(model,train,criterion,optimizer,device)
        vl,va=evaluate(model,val,criterion,device)
        for key,val1 in zip(history,(tl,ta,vl,va)):history[key].append(val1)
        if va>best:
            best=va;best_epoch=epoch+1
            torch.save(model.state_dict(),outdir/'best_model.pth')
        scheduler.step()
        print(f'Epoch {epoch+1:03d}: train_loss={tl:.4f}, train_acc={ta:.2f}, val_loss={vl:.4f}, val_acc={va:.2f}')
    model.load_state_dict(torch.load(outdir/'best_model.pth',map_location=device,weights_only=True))
    flip_scores={}
    for mode in ('original','flip'):
        flip_scores[mode]=evaluate(model,val,criterion,device,flip=mode=='flip')
    selected=max(flip_scores,key=lambda k:(flip_scores[k][1],-flip_scores[k][0]))
    test_loss,test_acc=evaluate(model,test,criterion,device,flip=selected=='flip')
    summary=dict(model=args.model,epochs=args.epochs,batch=args.batch_size,
                 best_epoch=best_epoch,best_val_acc=best,tta_method=selected,
                 val_tta=flip_scores,test_loss=test_loss,test_acc=test_acc,history=history)
    (outdir/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot_history(history,outdir/'training_plots.png')
    print('FINAL TEST:',test_loss,test_acc)

if __name__=='__main__': main()
