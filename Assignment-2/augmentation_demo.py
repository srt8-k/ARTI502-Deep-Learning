"""ARTI 502 Assignment 2 - Task 4, CIFAR-10 augmentation demonstrations.
Run: python Assignment-2/augmentation_demo.py
Saves results/augmentation_examples.png; test transform is deterministic.
"""
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)

def main():
    data=datasets.CIFAR10('./cifar_data',train=True,download=True,
                          transform=transforms.ToTensor())
    tensor,label=data[0]
    image=transforms.ToPILImage()(tensor)
    flip=transforms.RandomHorizontalFlip(p=1)(image)
    crop=transforms.RandomResizedCrop(32,scale=(.70,.95))(image)
    jitter=transforms.ColorJitter(brightness=.4,contrast=.4,saturation=.4)(image)
    rotation=transforms.RandomRotation(degrees=25)(image)
    noise=torch.clamp(tensor+torch.randn_like(tensor)*.08,0,1)
    mean=torch.tensor(MEAN)[:,None,None]
    std=torch.tensor(STD)[:,None,None]
    normalized=(tensor-mean)/std
    restored=(normalized*std+mean).clamp(0,1)
    examples=[
        ('Original',tensor),
        ('Horizontal flip',transforms.ToTensor()(flip)),
        ('Resized crop',transforms.ToTensor()(crop)),
        ('Color jitter',transforms.ToTensor()(jitter)),
        ('Random rotation',transforms.ToTensor()(rotation)),
        ('Gaussian noise',noise),
        ('Normalize (restored)',restored)]
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,(name,im) in zip(axes.flat,examples):
        ax.imshow(im.permute(1,2,0).numpy().clip(0,1))
        ax.set_title(name)
        ax.axis('off')
    for ax in axes.flat[len(examples):]:
        ax.axis('off')
    fig.suptitle(f'CIFAR-10 Data Augmentation Examples (class: {data.classes[label]})')
    fig.tight_layout()
    Path('results').mkdir(exist_ok=True)
    fig.savefig('results/augmentation_examples.png',dpi=180,bbox_inches='tight')
    plt.show()

if __name__=='__main__': main()
