"""EfficientNetV2-M CIFAR-10 fine-tuning (ARTI 502 Assignment 2).
CNN experiment based on the recorded Colab configuration.
Pretrained ImageNet weights; 45k/5k train/validation split; official test used once.
Requires: torch, torchvision, timm, matplotlib, numpy.
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset

MEAN=(.5,.5,.5)
STD=(.5,.5,.5)

def loaders(root,batch=16,seed=42):
    train_tf=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.ColorJitter(brightness=.1,contrast=.1,saturation=.1),
        transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    eval_tf=transforms.Compose([
        transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train=datasets.CIFAR10(root,train=True,download=True,transform=train_tf)
    val=datasets.CIFAR10(root,train=True,download=True,transform=eval_tf)
    test=datasets.CIFAR10(root,train=False,download=True,transform=eval_tf)
    indices=torch.randperm(50000,generator=torch.Generator().manual_seed(seed)).tolist()
    kw=dict(batch_size=batch,num_workers=2,pin_memory=torch.cuda.is_available())
    return (DataLoader(Subset(train,indices[:45000]),shuffle=True,**kw),
            DataLoader(Subset(val,indices[45000:]),shuffle=False,**kw),
            DataLoader(test,shuffle=False,**kw))

def cutmix(images,labels,alpha=1.0):
    batch,_,height,width=images.shape
    lam=float(torch.distributions.Beta(alpha,alpha).sample().item())
    shuffled=torch.randperm(batch,device=images.device)
    scale=math.sqrt(1-lam)
    crop_w=int(width*scale);crop_h=int(height*scale)
    cx=random.randrange(width);cy=random.randrange(height)
    x1=max(0,cx-crop_w//2);x2=min(width,cx+crop_w//2)
    y1=max(0,cy-crop_h//2);y2=min(height,cy+crop_h//2)
    result=images.clone()
    result[:,:,y1:y2,x1:x2]=images[shuffled,:,y1:y2,x1:x2]
    lam=1-((x2-x1)*(y2-y1))/(width*height)
    return result,labels,labels[shuffled],lam

@torch.inference_mode()
def evaluate(model,loader,device,tta='original'):
    model.eval()
    correct=total=0; loss_sum=0.
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)
        views=[images]
        if tta in ('flip','multiscale_flip'):
            views.append(images.flip(-1))
        if tta=='multiscale_flip':
            scaled=F.interpolate(images,size=(240,240),mode='bicubic',align_corners=False)
            views.extend([scaled,scaled.flip(-1)])
        probabilities=None
        for v in views:
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(v)
            p=logits.float().softmax(1)
            probabilities=p if probabilities is None else probabilities+p
        probabilities=probabilities/len(views)
        loss_sum+=F.nll_loss(probabilities.clamp_min(1e-12).log(),labels,reduction='sum').item()
        correct+=(probabilities.argmax(1)==labels).sum().item()
        total+=labels.size(0)
    return loss_sum/total,100*correct/total

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--data-root',default='./cifar_data')
    parser.add_argument('--output',default='./results/efficientnetv2m')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    assert torch.cuda.is_available(),'This model requires a CUDA GPU for practical training.'
    device=torch.device('cuda')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train,val,test=loaders(args.data_root,args.batch_size)
    model=timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k',pretrained=True,
                            num_classes=10,drop_rate=0.0,drop_path_rate=.2).to(device)
    head=list(model.get_classifier().parameters())
    head_ids={id(p) for p in head}
    backbone=[p for p in model.parameters() if id(p) not in head_ids]
    optimizer=torch.optim.AdamW([
        dict(params=backbone,lr=3e-5),
        dict(params=head,lr=3e-4)],weight_decay=.005)
    steps_per_epoch=math.ceil(len(train)/args.accumulation)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=[3e-5,3e-4],total_steps=args.epochs*steps_per_epoch,
        pct_start=.1,anneal_strategy='cos',div_factor=10,final_div_factor=100)
    criterion=nn.CrossEntropyLoss(label_smoothing=.05)
    scaler=torch.amp.GradScaler('cuda')
    history={k:[] for k in ('train_loss','train_acc','val_loss','val_acc')}
    best=-1.;best_epoch=0;start_epoch=0
    last_file=out/'last_checkpoint.pth';best_file=out/'best_weights.pth'
    if args.resume:
        checkpoint=torch.load(last_file,map_location='cpu',weights_only=True)
        assert checkpoint['epochs']==args.epochs and checkpoint['accumulation']==args.accumulation
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        history=checkpoint['history'];best=checkpoint['best'];best_epoch=checkpoint['best_epoch']
        start_epoch=checkpoint['epoch']
    for epoch in range(start_epoch,args.epochs):
        t0=time.time();model.train()
        correct=loss_sum=total=0.
        optimizer.zero_grad(set_to_none=True)
        for batch_index,(images,labels) in enumerate(train):
            images,labels=images.to(device),labels.to(device)
            if random.random()<.5:
                images,labels_a,labels_b,lam=cutmix(images,labels)
            else:
                labels_a=labels_b=labels;lam=1.
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(images)
                loss=lam*criterion(logits,labels_a)+(1-lam)*criterion(logits,labels_b)
            group_start=(batch_index//args.accumulation)*args.accumulation
            group_size=min(args.accumulation,len(train)-group_start)
            scaler.scale(loss/group_size).backward()
            if (batch_index+1)%args.accumulation==0 or batch_index+1==len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                old_scale=scaler.get_scale()
                scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=old_scale:scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            n=labels.size(0)
            total+=n;loss_sum+=loss.item()*n
            pred=logits.argmax(1)
            correct+=lam*(pred==labels_a).sum().item()+(1-lam)*(pred==labels_b).sum().item()
        train_loss=loss_sum/total
        train_acc=100*correct/total  # CutMix-weighted, not clean-image train accuracy
        val_loss,val_acc=evaluate(model,val,device)
        for k,v in zip(history,(train_loss,train_acc,val_loss,val_acc)):
            history[k].append(v)
        if val_acc>best:
            best=val_acc;best_epoch=epoch+1
            torch.save(model.state_dict(),best_file)
        torch.save(dict(epoch=epoch+1,epochs=args.epochs,accumulation=args.accumulation,
                        model=model.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                        best=best,best_epoch=best_epoch,history=history),last_file)
        print(f'Epoch {epoch+1}/{args.epochs}: train={train_acc:.2f}% (CutMix weighted), '
              f'val={val_acc:.2f}%, loss={val_loss:.4f}, best={best:.2f}%, '
              f'sec={time.time()-t0:.1f}',flush=True)
    model.load_state_dict(torch.load(best_file,map_location=device,weights_only=True))
    val_results={m:evaluate(model,val,device,tta=m) for m in ('original','flip','multiscale_flip')}
    selected=max(val_results,key=lambda m:(val_results[m][1],-val_results[m][0]))
    test_loss,test_accuracy=evaluate(model,test,device,tta=selected)
    results=dict(architecture='CNN EfficientNetV2-M',pretrained=True,
                 best_epoch=best_epoch,best_val_accuracy=best,
                 validation_tta=val_results,selected_tta=selected,
                 final_test_accuracy=test_accuracy,final_test_loss=test_loss,
                 train_subset=45000,val_subset=5000,test_subset=10000,history=history)
    (out/'experiment_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('FINAL TEST ACCURACY:',f'{test_accuracy:.2f}%','LOSS:',f'{test_loss:.4f}')

if __name__=='__main__':
    main()
